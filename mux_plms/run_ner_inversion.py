#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric, ClassLabel
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    PreTrainedTokenizerFast,
)
from datamux_pretraining.models.multiplexing_legacy import (
    RobertaTokenClassificationMuxed,
)
from datamux_pretraining.models.finetune_trainer import FinetuneTrainer
from datamux_pretraining.models.multiplexing_pretraining_electra import (
    MuxedElectraForTokenClassification,
)
from datamux_pretraining.models.multiplexing_pretraining_bert import (
    MuxedBertForTokenClassification,
)
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from rouge_score import rouge_scorer
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import random
import evaluate
from matplotlib import pyplot as plt
import json
import pickle

logger = logging.getLogger(__name__)

version_2_modelcls = {
    "electra": MuxedElectraForTokenClassification,
    "datamux_legacy": RobertaTokenClassificationMuxed,
    "bert": MuxedBertForTokenClassification,
}

class InversionPLM(nn.Module):
    def __init__(self, config, model_name_or_path='roberta-base'):
        super(InversionPLM, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, label, attention_mask=None):
        outputs = self.model(inputs_embeds=x, labels=label, attention_mask=attention_mask)
        return outputs.logits, outputs.loss

    def predict(self, x, label=None, attention_mask=None):
        outputs = self.model(inputs_embeds=x, labels=label, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(F.softmax(logits,dim=-1), dim=2)
        return logits, pred

def mux_token_selection(model, filter_tokens, batch, real_sentence_idx, dataset_word_dict, select_strategy='None',  token2cluster=None, clusters2token_list=None): 
    batch_size = batch['input_ids'].size()[0]
    emb = model.bert.embeddings.word_embeddings.weight
    dataset_word_dict = torch.tensor(dataset_word_dict)
    device = 'cuda'
    if select_strategy == 'similar' or select_strategy=='far':
        largest = True if select_strategy=='far' else False
        real_sentence_embedding = model.bert.embeddings(input_ids=batch['input_ids'][real_sentence_idx].unsqueeze(0))
        emb_dataset = emb[dataset_word_dict]
        ed = torch.cdist(real_sentence_embedding, emb_dataset, p=2.0) # (samples, embeddings)
        candidate_token_ids_top100_idx = torch.topk(ed, 100, largest=largest)[1] # (samples, topk)
        candidate_token_ids_top100 = dataset_word_dict[candidate_token_ids_top100_idx.view(-1)]
        candidate_token_ids_top100 = candidate_token_ids_top100.view(candidate_token_ids_top100_idx.size()[0], candidate_token_ids_top100_idx.size()[1], -1)
        candidate_token_ids_top100 = candidate_token_ids_top100.repeat(batch_size,1,1).to('cuda')
        # 筛选出要替换的词
        invalid_ids = (batch['attention_mask'] == 0)
        invalid_ids[word_filter(batch['input_ids'], filter_tokens)] = True
        # 真实句子的词不进行替换操作
        invalid_ids[real_sentence_idx] = False
        # 只对假句子长度不足的情况进行替换操作
        real_sentence_length = list(batch['input_ids'][real_sentence_idx]).index(102)
        invalid_ids[:,real_sentence_length:] = False
        # 生成待替换词的随机下标
        selection_ids = torch.randint(low=1, high=100, size=batch['input_ids'][invalid_ids].size()).unsqueeze(1).to('cuda')
        selection_tokens = torch.gather(input=candidate_token_ids_top100[invalid_ids], dim=1, index=selection_ids)
        batch['input_ids'][invalid_ids] = selection_tokens.squeeze(1)
    elif select_strategy == 'random':
        # 筛选出要替换的词
        invalid_ids = (batch['attention_mask'] == 0)
        invalid_ids[word_filter(batch['input_ids'], filter_tokens)] = True
        # 真实句子的词不进行替换操作
        invalid_ids[real_sentence_idx] = False
        # 只对假句子长度不足的情况进行替换操作
        real_sentence_length = list(batch['input_ids'][real_sentence_idx]).index(102)
        invalid_ids[:,real_sentence_length:] = False
        # 生成待替换词的随机下标
        selection_ids = torch.randint(low=0, high=len(dataset_word_dict)-1, size=batch['input_ids'][invalid_ids].size())
        selection_tokens = dataset_word_dict[selection_ids].to('cuda')
        batch['input_ids'][invalid_ids] = selection_tokens
    elif select_strategy == 'all_random':
        # 筛选出要替换的词
        invalid_ids = (batch['attention_mask'] == 0)
        invalid_ids[:,1:] = True
        # 真实句子的词不进行替换操作
        invalid_ids[real_sentence_idx] = False
        # 只对假句子长度不足的情况进行替换操作
        real_sentence_length = list(batch['input_ids'][real_sentence_idx]).index(102)
        invalid_ids[:,real_sentence_length:] = False
        # 生成待替换词的随机下标
        selection_ids = torch.randint(low=0, high=len(dataset_word_dict)-1, size=batch['input_ids'][invalid_ids].size())
        selection_tokens = dataset_word_dict[selection_ids].to('cuda')
        batch['input_ids'][invalid_ids] = selection_tokens
    elif select_strategy == 'all_similar' or select_strategy=='all_far':
        largest = True if select_strategy=='far' else False
        real_sentence_embedding = model.bert.embeddings(input_ids=batch['input_ids'][real_sentence_idx].unsqueeze(0))
        emb_dataset = emb[dataset_word_dict]
        ed = torch.cdist(real_sentence_embedding, emb_dataset, p=2.0) # (samples, embeddings)
        candidate_token_ids_top100_idx = torch.topk(ed, 100, largest=largest)[1] # (samples, topk)
        candidate_token_ids_top100 = dataset_word_dict[candidate_token_ids_top100_idx.view(-1)]
        candidate_token_ids_top100 = candidate_token_ids_top100.view(candidate_token_ids_top100_idx.size()[0], candidate_token_ids_top100_idx.size()[1], -1)
        candidate_token_ids_top100 = candidate_token_ids_top100.repeat(batch_size,1,1).to('cuda')
        # 筛选出要替换的词
        invalid_ids = (batch['attention_mask'] == 0)
        invalid_ids[:,1:] = True
        # 真实句子的词不进行替换操作
        invalid_ids[real_sentence_idx] = False
        # 只对假句子长度不足的情况进行替换操作
        real_sentence_length = list(batch['input_ids'][real_sentence_idx]).index(102)
        invalid_ids[:,real_sentence_length:] = False
        # 生成待替换词的随机下标
        selection_ids = torch.randint(low=1, high=100, size=batch['input_ids'][invalid_ids].size()).unsqueeze(1).to('cuda')
        selection_tokens = torch.gather(input=candidate_token_ids_top100[invalid_ids], dim=1, index=selection_ids)
        batch['input_ids'][invalid_ids] = selection_tokens.squeeze(1)
    elif select_strategy=="cluster":
        real_sentence_length = list(batch['input_ids'][real_sentence_idx]).index(102)
        real_sentence = batch['input_ids'][real_sentence_idx]
        real_sentence_sample_num = 4 # [0, num_instances-1] or (batch_size-1) // 2
        real_sentence_sample_pool_repeat = (real_sentence_sample_num // (real_sentence_length-1)) + 1
        for idx in range(1, real_sentence_length):
            cur_token = real_sentence[idx]
            # real sentence sample
            real_sentence_sample_pool = list(real_sentence[1:real_sentence_length].repeat(real_sentence_sample_pool_repeat))
            real_sentence_sample_pool.remove(cur_token)
            real_sentence_selected_tokens = random.sample(real_sentence_sample_pool, k=real_sentence_sample_num)
            # cluster sample
            cluster_id = token2cluster[int(cur_token.item())]
            cluster_sample_pool = clusters2token_list[cluster_id]
            # cluster_sample_pool.remove(int(cur_token.item()))
            if len(cluster_sample_pool) < batch_size-1-real_sentence_sample_num:
                cluster_sample_pool+=real_sentence_sample_pool
            cluster_selected_tokens = random.sample(cluster_sample_pool, k=batch_size-1-real_sentence_sample_num)
            selected_tokens = cluster_selected_tokens + real_sentence_selected_tokens
            random.shuffle(selected_tokens)
            selected_tokens.insert(real_sentence_idx, cur_token)
            selected_tokens = torch.tensor(selected_tokens)
            batch['input_ids'][:,idx] = selected_tokens
    else:
        pass
    return batch
    
def dataloader2memory(dataloader, model, tokenizer, num_instances, dataset_word_dict, select_strategy, dataset_name, target_layer=3, device='cuda'):
    token_shuffle = True
    token2cluster = None
    clusters2token_list = None
    features = []
    pro_bar = tqdm(range(len(dataloader)))
    model.eval()
    # filter special tokens
    special_tokens = tokenizer.convert_tokens_to_ids(['[PAD]','[SEP]'])
    simple_tokens = tokenizer.convert_tokens_to_ids(['.', ',', '"', '-',"'",'(',')',':','the','a','t','n','?','%','of','and','s','to','it','is','was','for','that','in','as','on'])
    filter_tokens = list(set(special_tokens + simple_tokens))
    if select_strategy == "conll_mux":
        print('get conll2003 dataset sentences')
        conll2003_sentences = get_conll2003_sentences(tokenizer)
        print('done')
    elif select_strategy == "cluster":
        dataset_word_num = len(dataset_word_dict)
        n_clusters = dataset_word_num // (num_instances * 10)
        # save result
        cluster_dir = f'kmeans_cluster_num{n_clusters}'
        cluster_path = os.path.join(f'/root/mixup/mux_plms/cluster/{dataset_name}', cluster_dir)
        with open(f'{cluster_path}/{cluster_dir}_token2cluster.pickle',"rb") as f:
            token2cluster = pickle.load(f)
        with open(f'{cluster_path}/{cluster_dir}_clusters2token_list.pickle',"rb") as f:
            clusters2token_list = pickle.load(f)
        
    for batch in dataloader:
        with torch.no_grad():
            if 'idx' in batch.keys():
                batch.pop("idx")
            batch = {key:value.to(device) for key,value in batch.items()}
            batch_size, sequence_length = batch["input_ids"].size()
            all_hidden_states = [] 
            all_mux_sentence_input_ids = []
            for idx in range(batch_size):
                sample_list = list(range(0,batch_size))
                sample_list.remove(idx)
                mux_sentence_ids = random.sample(sample_list, k=num_instances-1)
                # rand real sentence pos
                real_sentence_idx = random.randint(0, len(mux_sentence_ids))
                mux_sentence_ids.insert(real_sentence_idx, idx)
                # fix real sentence pos
                # real_sentence_idx = len(mux_sentence_ids)
                # mux_sentence_ids.insert(real_sentence_idx, idx)
                
                mux_minibatch = {key:value[mux_sentence_ids] for key,value in batch.items()}
                if select_strategy == "conll_mux":
                    sample_list = list(range(0,len(conll2003_sentences)))
                    sample_sentence_idx = random.sample(sample_list, k=num_instances-1)
                    sample_sentences = [conll2003_sentences[idx] for idx in sample_sentence_idx]
                    sample_sentences = torch.tensor(sample_sentences).to(device)
                    replaced_idx = list(range(num_instances))
                    replaced_idx.remove(real_sentence_idx)
                    mux_minibatch['input_ids'][replaced_idx] = sample_sentences
                mux_minibatch = mux_token_selection(model, filter_tokens, mux_minibatch, real_sentence_idx, dataset_word_dict, select_strategy, token2cluster, clusters2token_list)
                mux_minibatch['real_sentence_idx'] = real_sentence_idx
                outputs = model(**mux_minibatch)
                hidden_states = outputs.hidden_states
                all_mux_sentence_input_ids.append(mux_minibatch['input_ids'])
                all_hidden_states.append(hidden_states)
            input_ids = batch['input_ids'].to('cpu')
            attention_mask = batch['attention_mask'].to('cpu')
            target_hidden_states = torch.cat(all_hidden_states, dim=0).to('cpu')
            # target_hidden_states = torch.stack(all_hidden_states).to('cpu')
            all_mux_sentence_input_ids = torch.stack(all_mux_sentence_input_ids)
            features.append({'hidden_states': target_hidden_states, 'input_ids': input_ids, 'attention_mask': attention_mask, 'mux_sentence_input_ids': all_mux_sentence_input_ids})
        pro_bar.update(1)
    return features

def word_filter(eval_label, filter_list):
    allow_token_ids = (eval_label == filter_list[0])
    for item in filter_list:
        allow_token_ids = allow_token_ids | (eval_label == item)
    return allow_token_ids

def get_dataset_word_dict(train_dataset, eval_dataset):
    word_set = set()
    for sentence in train_dataset['input_ids']:
        for token in sentence:
            word_set.add(token)
    for sentence in eval_dataset['input_ids']:
        for token in sentence:
            word_set.add(token)
    word_set = sorted(list(word_set))
    return word_set

def dataset_statistic(train_dataloader, eval_dataloader, dataset_name, tokenizer):
    stat_result_path = f'/root/mixup/mux_plms/dataset_stat/{dataset_name}'
    if not os.path.exists(stat_result_path):
        os.makedirs(stat_result_path)
    tokens_dict = {}
    sentence_length_list = []
    for batch in tqdm(train_dataloader):
        batch_size, seq_length = batch['input_ids'].size()
        for seq_idx in range(batch_size):
            sentence_length = list(batch['input_ids'][seq_idx]).index(102)
            sentence_length_list.append(sentence_length)
            for word_idx in range(seq_length):
                token = tokenizer.decode(batch['input_ids'][seq_idx][word_idx])
                tokens_dict[token] = tokens_dict.get(token, 0) + 1
    for batch in tqdm(eval_dataloader):
        batch_size, seq_length = batch['input_ids'].size()
        for seq_idx in range(batch_size):
            sentence_length = list(batch['input_ids'][seq_idx]).index(102)
            sentence_length_list.append(sentence_length)
            for word_idx in range(seq_length):
                token = tokenizer.decode(batch['input_ids'][seq_idx][word_idx])
                tokens_dict[token] = tokens_dict.get(token, 0) + 1
    tokens_dict = sorted(tokens_dict.items(), key=lambda x: x[1], reverse=True)
    with open(os.path.join(stat_result_path, f'token_stat.txt'),'w') as f:
        token_num = len(tokens_dict)
        f.write(f'dataset total tokens: {token_num}\n')
        for (key, value) in tokens_dict:
            f.write(f'{key:<15s}: {value}\n')
    with open(os.path.join(stat_result_path, f'sentence_stat.txt'),'w') as f:
        seq_mean_length = sum(sentence_length_list) / len(sentence_length_list)
        f.write(f'dataset mean sequence length: {seq_mean_length}\n')
        for seq_len in sentence_length_list:
            f.write(f'{seq_len}\n')
        

def train_inversion_model(config, tokenizer, model, train_dataloader, eval_dataloader, dataset_word_dict, use_wandb=True):
    # batch_size=32 # {roberta:32, mlp:64}
    learning_rate=2e-5 # {roberta:1e-5 2e-5 5e-5, mlp:2e-4}
    device='cuda'
    epochs=30
    topk = 1
    inversion_model = InversionPLM(config)

    inversion_model = inversion_model.to(device)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(inversion_model.parameters(), lr=learning_rate)
    
    print('load dataloader to memory')
    train_dataloader = dataloader2memory(train_dataloader, model, tokenizer, config.num_instances, dataset_word_dict, config.select_strategy, config.dataset_name, config.target_layer, device)
    eval_dataloader = dataloader2memory(eval_dataloader, model, tokenizer, config.num_instances, dataset_word_dict, config.select_strategy, config.dataset_name, config.target_layer, device)
    print('done')
    
    total_step = len(train_dataloader) * epochs
    
    progress_bar = tqdm(range(total_step))
    special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    # filted inversion
    # simple_tokens = tokenizer.convert_tokens_to_ids(['.', ',', '"', '-',"'",'(',')',':','the','a','t','n','?','%','of','and','s','to','is','was','for','that','in','as','on'])
    # origin inversion
    simple_tokens = tokenizer.convert_tokens_to_ids(['.', ',', '"', '-'])
    filter_tokens = list(set(special_tokens + simple_tokens))
    completed_steps = 0
    model_attack_acc = 0
    hit_tokens = {}
    mux_tokens_list = []
    print('################# start train inversion model #################')
    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            batch = {key:value.to(device) for key,value in batch.items()}
            target_hidden_states = batch['hidden_states']
            labels = batch['input_ids']
            labels[labels == tokenizer.pad_token_id] = -100
            
            attention_mask = batch['attention_mask']
            
            bsz, seq_len, dim = target_hidden_states.shape
            feature = target_hidden_states
            
            feature = feature.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            logits, loss = inversion_model(feature, labels, attention_mask=attention_mask)
            if use_wandb:
                wandb.log({'loss/inversion_model_loss':loss.item()})

            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
            progress_bar.update(1)
            progress_bar.set_description('inversion_model_loss:{}'.format(loss.item()))

        with torch.no_grad():
            hit_cnt = 0
            total_cnt = 0
            case_study_dir = f'{config.wandb_name}'
            case_study_path = os.path.join(f'/root/mixup/mux_plms/case_study/{config.task_name}/mux_{config.num_instances}',case_study_dir)
            if not os.path.exists(case_study_path):
                os.makedirs(case_study_path)
            with open(os.path.join(case_study_path, f'inversion_epoch{epoch}.txt'),'w') as f:
                for batch in eval_dataloader:
                    batch = {key:value.to(device) for key,value in batch.items()}
                    # batch['output_hidden_states'] = True
                    
                    # outputs = model(**batch)
                    # target_hidden_states = outputs.hidden_states[target_layer]
                    target_hidden_states = batch['hidden_states']
                    eval_label = batch['input_ids']
                    attention_mask = batch['attention_mask']

                    bsz, seq_len, dim = target_hidden_states.shape
                    feature = target_hidden_states
                    feature = feature.to(device)
                    attention_mask = attention_mask.to(device)

                    # feature = torch.cat([feature[:, 0], feature[:, 1]], dim=2)
                    
                    pred_logits, preds = inversion_model.predict(feature, attention_mask=attention_mask)

                    valid_ids = attention_mask!=0
                    
                    
                    valid_ids[word_filter(eval_label, filter_tokens)] = False
                    eval_label = batch['input_ids']
                    eval_label = eval_label[valid_ids] 
                    preds = torch.topk(pred_logits, k=topk)[1]
                    preds = preds[valid_ids]
                    hit_cnt += (eval_label.unsqueeze(1) == preds).int().sum().item()
                    total_cnt += eval_label.shape[0]
                    # 进行case_study记录
                    top10_preds = torch.topk(pred_logits, k=10)[1]
                    for seq_idx in range(0,  batch['input_ids'].size()[0]):
                        f.write(f'-----------------------------muxplm-{config.task_name}-{config.num_instances} case start-----------------------------\n')
                        # titile
                        f.write(f"{'origin_token':<20s} | ")
                        for mux_idx in range(config.num_instances):
                            f.write(f"{f'mux_token{mux_idx}':<20s}")
                        f.write(" | ")
                        for rec_idx in range(config.num_instances):
                            f.write(f"{f'recover_token{rec_idx}':<20s}")
                        f.write('\n')
                        # f.write(f"{'mux_token1':<20s}{'mux_token2':<20s}{'mux_token3':<20s}{'mux_token4':<20s}{'mux_token5':<20s}{'mux_token6':<20s}{'mux_token7':<20s}{'mux_token8':<20s}{'mux_token9':<20s}{'mux_token10':<20s} | ")
                        # f.write(f"{'recover_token1':<20s}{'recover_token2':<20s}{'recover_token3':<20s}{'recover_token4':<20s}{'recover_token5':<20s}{'recover_token6':<20s}{'recover_token7':<20s}{'recover_token8':<20s}{'recover_token9':<20s}{'recover_token10':<20s}")
                        for word_idx in range(0, batch['input_ids'].size()[1]):
                            if valid_ids[seq_idx][word_idx]:
                                f.write(f"{tokenizer.decode(batch['input_ids'][seq_idx][word_idx]):<20s} | ")
                                for mux_idx in range(config.num_instances):
                                    f.write(f"{tokenizer.decode(batch['mux_sentence_input_ids'][seq_idx][mux_idx][word_idx]):<20s}")
                                f.write(" | ")
                                for rec_idx in range(config.num_instances):
                                    f.write(f"{tokenizer.decode(top10_preds[seq_idx][word_idx][rec_idx]):<20s}")
                                # if token hited in last epoch
                                if epoch == epochs - 1: 
                                    if batch['input_ids'][seq_idx][word_idx] == top10_preds[seq_idx][word_idx][0]:
                                        hit_tokens[tokenizer.decode(batch['input_ids'][seq_idx][word_idx])] = hit_tokens.get(tokenizer.decode(batch['input_ids'][seq_idx][word_idx]), 0) + 1
                                        mux_tokens = [tokenizer.decode(batch['mux_sentence_input_ids'][seq_idx][mux_idx][word_idx]) for mux_idx in range(config.num_instances)]
                                        mux_tokens_list.append(mux_tokens)
                                f.write("\n")
                        f.write(f'-----------------------------muxplm-{config.task_name}-{config.num_instances} case end-----------------------------\n')
                        f.write('\n')
            model_attack_acc = hit_cnt/total_cnt
            print('attack acc:{}'.format(hit_cnt/total_cnt))
            if use_wandb:
                wandb.log({'metric/inversion_model_top{}_acc'.format(topk): hit_cnt/total_cnt})
    
    with open(os.path.join(case_study_path, 'hit_tokens_stat.txt'),'w') as f:
        hit_tokens = sorted(hit_tokens.items(), key=lambda x: x[1], reverse=True)
        for (key, value) in hit_tokens:
            f.write(f'{key:<15s}: {value}\n')
    with open(os.path.join(case_study_path, 'mux_tokens_stat.txt'),'w') as f:
        for mux_tokens in mux_tokens_list:
            for mux_token in mux_tokens:
                f.write(f"{mux_token:<15s}")
            f.write('\n')
    with open(os.path.join(case_study_path, 'mux_tokens_conbine_stat.txt'),'w') as f:
        mux_conbination = {}
        for mux_tokens in mux_tokens_list:
            conbination = tuple(set(mux_tokens))
            mux_conbination[conbination] = mux_conbination.get(conbination, 0) + 1
        mux_conbination = sorted(mux_conbination.items(), key=lambda x: x[1], reverse=True)
        for (key, value) in mux_conbination:
            f.write(f"conbination_times : {value}\t")
            f.write(f"conbination_tokens : ")
            for mux_token in key:
                f.write(f"{mux_token:<15s}")
            f.write('\n')
    # torch.save(model, 'mux_plms/ckpts/sst2/datamux-sst2-10/inversion_model_token_shuffle.pt')
    return model_attack_acc


def rouge(input_ids, pred_ids, tokenizer):
    # input_ids (bsz, seq_len)
    # pred_ids (bsz, seq_len)
    batch_real_tokens = [tokenizer.decode(item, skip_special_tokens=True) for item in input_ids]
    batch_pred_tokens = [tokenizer.decode(item, skip_special_tokens=True) for item in pred_ids]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    # scores = scorer.score('The quick brown fox jumps over the lazy dog',
    #                   'The quick brown dog jumps on the log.')
    hit_cnt = 0
    total_cnt = 0
    for real_tokens, pred_tokens in zip(batch_real_tokens, batch_pred_tokens):
        rouge_score = scorer.score(real_tokens, pred_tokens)['rougeL'].fmeasure
        hit_cnt += rouge_score
        total_cnt += 1
    return hit_cnt, total_cnt

def compute_metrics(metric, return_entity_level_metrics=False):
        results = metric.compute()
        if return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

def get_labels(predictions, references, label_list):
        # Transform predictions and references tensos to numpy arrays
        if predictions.device == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

def evaluate_with_knn_attack(model, dataloader, tokenizer, metric, config, label_list, dataset_word_dict):
    emb = model.bert.embeddings.word_embeddings.weight
    device = 'cuda'
    select_strategy = config.select_strategy
    token2cluster = None
    clusters2token_list = None
    if select_strategy == "conll_mux":
        conll2003_sentences = get_conll2003_sentences(tokenizer)
    elif select_strategy == "cluster":
        dataset_word_num = len(dataset_word_dict)
        n_clusters = dataset_word_num // (config.num_instances * 10)
        # save result
        cluster_dir = f'kmeans_cluster_num{n_clusters}'
        cluster_path = os.path.join(f'/root/mixup/mux_plms/cluster/{config.dataset_name}', cluster_dir)
        with open(f'{cluster_path}/{cluster_dir}_token2cluster.pickle',"rb") as f:
            token2cluster = pickle.load(f)
        with open(f'{cluster_path}/{cluster_dir}_clusters2token_list.pickle',"rb") as f:
            clusters2token_list = pickle.load(f)
    model.eval()
    # hit
    top1_hit_cnt = 0
    top5_hit_cnt = 0
    top10_hit_cnt = 0
    rouge_hit_cnt = 0
    # total
    total_cnt = 0
    rouge_total_cnt = 0
    # filter special tokens for mux token selection
    special_tokens_mux_token_selection = tokenizer.convert_tokens_to_ids(['[PAD]','[SEP]'])
    simple_tokens_mux_token_selection = tokenizer.convert_tokens_to_ids(['.', ',', '"', '-',"'",'the','a','of','and','s','to','it','is','that','in','as','on','(',')'])
    filter_tokens_mux_token_selection = list(set(special_tokens_mux_token_selection + simple_tokens_mux_token_selection))
    # filter special tokens for knn attack
    special_tokens_knn_attack = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    simple_tokens_knn_attack = tokenizer.convert_tokens_to_ids(['.', ',', '"', '-'])
    filter_tokens_knn_attack = list(set(special_tokens_knn_attack + simple_tokens_knn_attack))
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = {key:value.to(device) for key,value in batch.items()}
            batch_size, sequence_length = batch["input_ids"].size()
            all_predictions = []
            all_hidden_states = []
            for idx in range(batch_size):
                sample_list = list(range(0,batch_size))
                sample_list.remove(idx)
                mux_sentence_ids = random.sample(sample_list, k=config.num_instances-1)
                # rand real sentence pos
                real_sentence_idx = random.randint(0, len(mux_sentence_ids))
                mux_sentence_ids.insert(real_sentence_idx, idx)
                # fix real sentence pos
                # real_sentence_idx = len(mux_sentence_ids)
                # mux_sentence_ids.insert(real_sentence_idx, idx)
                
                mux_minibatch = {key:value[mux_sentence_ids] for key,value in batch.items()}
                if select_strategy == "conll_mux":
                    sample_list = list(range(0,len(conll2003_sentences)))
                    sample_sentence_idx = random.sample(sample_list, k=config.num_instances-1)
                    sample_sentences = [conll2003_sentences[idx] for idx in sample_sentence_idx]
                    sample_sentences = torch.tensor(sample_sentences).to(device)
                    replaced_idx = list(range(config.num_instances))
                    replaced_idx.remove(real_sentence_idx)
                    mux_minibatch['input_ids'][replaced_idx] = sample_sentences
                mux_minibatch = mux_token_selection(model, filter_tokens_mux_token_selection, mux_minibatch, real_sentence_idx, dataset_word_dict, select_strategy, token2cluster, clusters2token_list)
                mux_minibatch['real_sentence_idx'] = real_sentence_idx 
                outputs = model(**mux_minibatch)
                hidden_states = outputs.hidden_states
                predictions = outputs.logits.argmax(dim=-1)
                all_predictions.append(predictions[real_sentence_idx])
                all_hidden_states.append(hidden_states)
            
        # evaluate
        all_predictions = torch.stack(all_predictions)
        all_hidden_states = torch.cat(all_hidden_states, dim=0)
        # all_hidden_states = torch.stack(all_hidden_states)
        labels = batch["labels"]
        preds, refs = get_labels(all_predictions, labels, label_list)
        metric.add_batch(
            predictions=preds,
            references=refs,
        )
        
        attention_mask = batch['attention_mask']
        valid_ids = attention_mask!=0  
        eval_label = batch['input_ids']
        valid_ids[word_filter(eval_label, filter_tokens_knn_attack)] = False
        eval_label = eval_label[valid_ids] # (samples)
        preds_feature = all_hidden_states[valid_ids]
        ed = torch.cdist(preds_feature, emb, p=2.0) # (samples, embeddings)
        candidate_token_ids_top1 = torch.topk(ed, 1, largest=False)[1] # (samples, topk)
        candidate_token_ids_top5 = torch.topk(ed, 5, largest=False)[1] # (samples, topk)
        candidate_token_ids_top10 = torch.topk(ed, 10, largest=False)[1] # (samples, topk)
        top1_hit_cnt += (eval_label.unsqueeze(1) == candidate_token_ids_top1).int().sum().item()
        top5_hit_cnt += (eval_label.unsqueeze(1) == candidate_token_ids_top5).int().sum().item()
        top10_hit_cnt += (eval_label.unsqueeze(1) == candidate_token_ids_top10).int().sum().item()
        total_cnt += eval_label.shape[0]
        # rouge
        r_hit_cnt, r_total_cnt = rouge(eval_label.unsqueeze(1), candidate_token_ids_top1, tokenizer)
        rouge_hit_cnt += r_hit_cnt
        rouge_total_cnt += r_total_cnt
    # compute metric
    eval_metric = compute_metrics(metric)
    eval_metric['knn_top1'] = top1_hit_cnt/total_cnt
    eval_metric['knn_top5'] = top5_hit_cnt/total_cnt
    eval_metric['knn_top10'] = top10_hit_cnt/total_cnt
    eval_metric['knn_rouge'] = rouge_hit_cnt/rouge_total_cnt
    return eval_metric

def get_conll2003_sentences(tokenizer):
    raw_datasets = load_dataset('conll2003')
    text_column_name = 'tokens'
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding='max_length',
            truncation=True,
            max_length=128,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        return tokenized_inputs

    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        load_from_cache_file=False,
    )
    return train_dataset['input_ids']

def cluster_pipeline(model, dataset_word_dict, dataset_name, num_instances):
    # get dataset embedding
    emb = model.bert.embeddings.word_embeddings.weight.cpu().detach().numpy()
    dataset_emb = emb[dataset_word_dict]
    dataset_word_num = len(dataset_word_dict)
    # clustering
    print('clustering......')
    import time
    start_time = time.time()
    n_clusters = dataset_word_num // (num_instances * 10)
    clusters = KMeans(n_clusters=n_clusters, random_state=0).fit(dataset_emb)
    dataset_emb_cluster_result = clusters.predict(dataset_emb)
    print('clustering......done! cost {} seconds'.format(time.time()-start_time))
    token2cluster = {}
    clusters2token_list = {}
    for dataset_token_ids, cluster_ids in enumerate(dataset_emb_cluster_result):
        cluster_ids = int(cluster_ids.item())
        token_input_ids = dataset_word_dict[dataset_token_ids]
        token2cluster[token_input_ids] = cluster_ids
        if cluster_ids not in clusters2token_list:
            clusters2token_list[cluster_ids] = []
        clusters2token_list[cluster_ids].append(token_input_ids)
    # save result 
    cluster_dir = f'kmeans_cluster_num{n_clusters}'
    cluster_path = os.path.join(f'/root/mixup/mux_plms/cluster/{dataset_name}', cluster_dir)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)        
    unique, counts = np.unique(list(token2cluster.values()), return_counts=True)
    plt.figure(figsize=(25, 10))
    plt.bar(unique, counts)
    plt.title(f'{cluster_dir} result statistics')
    plt.xlabel('cluster center index')
    plt.ylabel('num of tokens')
    plt.savefig(f'{cluster_path}/{cluster_dir}_vis.png')
    with open(f'{cluster_path}/{cluster_dir}_token2cluster.pickle',"wb") as f:
        pickle.dump(token2cluster,f)
    with open(f'{cluster_path}/{cluster_dir}_clusters2token_list.pickle',"wb") as f:
        pickle.dump(clusters2token_list,f)

    
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default="ner", metadata={"help": "The name of the task (ner, pos...)."}
    )

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )
    text_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The column name of text to input in the file (a csv or JSON file)."
        },
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The column name of label to input in the file (a csv or JSON file)."
        },
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={
            "help": "Whether to return all the entity levels during evaluation or just the overall ones."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    # multi instance arguments
    num_instances: Optional[int] = field(
        default=5,
        metadata={"help": "Number of instances i.e. N"},
    )
    muxing_variant: Optional[str] = field(
        default="gaussian_hadamard",
        metadata={
            "help": "muxing variant; choose from gaussian_hadamard or random_ortho or binary_hadamard"
        },
    )
    demuxing_variant: Optional[str] = field(
        default="index",
        metadata={"help": "demuxing variant, choose from  'index' or 'mlp'"},
    )
    should_mux: Optional[int] = field(
        default=1,
        metadata={"help": "whether to mux, turn off for non-multiplexed baselines"},
    )
    retrieval_percentage: Optional[float] = field(
        default=1.0,
        metadata={"help": "percentage of tokens to retrieve during inference"},
    )
    retrieval_pretraining: Optional[int] = field(
        default=0,
        metadata={"help": "Retrieval Pretraining"},
    )
    gaussian_hadamard_norm: Optional[float] = field(
        default=1,
        metadata={"help": "Norm of sentence embeddings if we use random projections"},
    )
    binary_hadamard_epsilon: Optional[float] = field(
        default=0,
        metadata={
            "help": "Percentage intersection among binary vectors, default is no intersection"
        },
    )
    retrieval_loss_coeff: Optional[float] = field(
        default=0.1,
        metadata={"help": "Coefficient for retrieval loss"},
    )
    task_loss_coeff: Optional[float] = field(
        default=0.9,
        metadata={"help": "Coefficient for task loss"},
    )
    learn_muxing: Optional[int] = field(
        default=0,
        metadata={"help": "whether instance embeddings are learnt or not"},
    )
    model_version: Optional[str] = field(
        default="bert",
        metadata={
            "help": "pretraining architecture, choose from  'roberta' or 'electra'"
        },
    )
    num_hidden_demux_layers: Optional[int] = field(
        default=3,
        metadata={"help": "number of hidden layers for demuxing"},
    )
    epsilon: Optional[float] = field(
        default=1.0,
        metadata={"help": "the scale for laplace noise"},
    )
    wandb_name: Optional[str] = field(
        default="mux_plm_default_wandb_name",
    )
    select_strategy: Optional[str] = field(
        default="None",
    )
    use_wandb: bool = field(
        default=False,
    )
    add_embedding_noise: bool = field(
        default=False,
    )
    eval_with_knn_attack: bool = field(
        default=False,
    )
    train_inversion_model: bool = field(
        default=False,
    )
    do_dataset_statistic: bool = field(
        default=False,
    )
    do_cluster: bool = field(
        default=False,
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension, data_files=data_files, cache_dir=model_args.cache_dir
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # if training_args.do_train:
    column_names = raw_datasets["train"].column_names
    features = raw_datasets["train"].features
    # else:
    #     column_names = raw_datasets["validation"].column_names
    #     features = raw_datasets["validation"].features

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
        
    if 'ontonotes' in data_args.dataset_name:
        label_to_id = {
        "O": 0,
        "B-CARDINAL": 1,
        "B-DATE": 2,
        "I-DATE": 3,
        "B-PERSON": 4,
        "I-PERSON": 5,
        "B-NORP": 6,
        "B-GPE": 7,
        "I-GPE": 8,
        "B-LAW": 9,
        "I-LAW": 10,
        "B-ORG": 11,
        "I-ORG": 12, 
        "B-PERCENT": 13,
        "I-PERCENT": 14, 
        "B-ORDINAL": 15, 
        "B-MONEY": 16, 
        "I-MONEY": 17, 
        "B-WORK_OF_ART": 18, 
        "I-WORK_OF_ART": 19, 
        "B-FAC": 20, 
        "B-TIME": 21, 
        "I-CARDINAL": 22, 
        "B-LOC": 23, 
        "B-QUANTITY": 24, 
        "I-QUANTITY": 25, 
        "I-NORP": 26, 
        "I-LOC": 27, 
        "B-PRODUCT": 28, 
        "I-TIME": 29, 
        "B-EVENT": 30,
        "I-EVENT": 31,
        "I-FAC": 32,
        "B-LANGUAGE": 33,
        "I-PRODUCT": 34,
        "I-ORDINAL": 35,
        "I-LANGUAGE": 36
    }
        label_list = list(label_to_id.keys())
    num_labels = len(label_list)

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label={i: l for l, i in label_to_id.items()},
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.num_instances = model_args.num_instances
    config.muxing_variant = model_args.muxing_variant
    config.demuxing_variant = model_args.demuxing_variant
    config.retrieval_percentage = model_args.retrieval_percentage
    config.gaussian_hadamard_norm = model_args.gaussian_hadamard_norm
    config.binary_hadamard_epsilon = model_args.binary_hadamard_epsilon
    config.retrieval_loss_coeff = model_args.retrieval_loss_coeff
    config.task_loss_coeff = model_args.task_loss_coeff
    config.learn_muxing = model_args.learn_muxing
    config.num_hidden_demux_layers = model_args.num_hidden_demux_layers
    config.add_embedding_noise = model_args.add_embedding_noise
    config.epsilon = model_args.epsilon
    config.target_layer = 3
    config.wandb_name = model_args.wandb_name
    config.task_name = data_args.task_name
    config.dataset_name = data_args.dataset_name
    config.select_strategy = model_args.select_strategy
    
    tokenizer_name_or_path = (
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path
    )

    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            # force_download=True,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model_path_supplied = model_args.model_name_or_path is not None
    if model_args.should_mux:
        model_cls = version_2_modelcls[model_args.model_version]
        if model_path_supplied:
            model = model_cls.from_pretrained(
                model_args.model_name_or_path,
                config=config,
            )
        else:
            model = model_cls(config=config)
    else:
        # non-multiplexed baseline
        if model_path_supplied:
            model = AutoModelForTokenClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
            )
        else:
            model = AutoModelForTokenClassification.from_config(config)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            if 'ontonotes' in data_args.dataset_name:
                label = [model.config.id2label[item] for item in label]
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if data_args.label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # if training_args.do_train:
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))

    train_dataset = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=False,
    )

    # if training_args.do_eval:
    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = raw_datasets["validation"]
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
    eval_dataset = eval_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=False,
    )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(
                range(data_args.max_predict_samples)
            )
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # # Metrics
    # metric = load_metric("seqeval")
    
    # Metrics
    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
    # use wandb
    if model_args.use_wandb:
        project_name = f'muxplm_{data_args.dataset_name}_mux{config.num_instances}' if 'ontonotes' not in data_args.dataset_name else f'muxplm_ontonotes_mux{config.num_instances}'
        wandb.init(config=config, project=project_name, entity='privacy_cluster', name=model_args.wandb_name, sync_tensorboard=False,
                job_type="CleanRepo", settings=wandb.Settings(start_method='fork'))
    # Initialize our Trainer
    trainer = FinetuneTrainer(
        model=model,
        args=training_args,
        # train_dataset=train_dataset if training_args.do_train else None,
        # eval_dataset=eval_dataset if training_args.do_eval else None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        output_dir = training_args.output_dir
        last_checkpoint = get_last_checkpoint(output_dir)
        logger.info(f"Loading model from {last_checkpoint}")
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        all_metrics = {}
        for seed in range(1, 6):
            set_seed(seed)
            metrics = trainer.evaluate(
                eval_dataset=eval_dataset, resume_from_checkpoint=checkpoint
            )
            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            for key, value in metrics.items():
                all_metrics[f"{key}_{seed}"] = value

        trainer.log_metrics("eval", all_metrics)
        trainer.save_metrics("eval", all_metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        )
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(
            training_args.output_dir, "predictions.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "token-classification",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=60, drop_last=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=60, drop_last=True)
    torch.cuda.empty_cache()
    
    print('statistic dataset word dict')
    dataset_word_dict = get_dataset_word_dict(train_dataset, eval_dataset)
    print('done')
    
    if model_args.do_dataset_statistic:
        dataset_statistic(train_dataloader, eval_dataloader, data_args.dataset_name, tokenizer)
    if model_args.do_cluster:
        cluster_pipeline(model, dataset_word_dict, data_args.dataset_name, config.num_instances)
    
    if model_args.eval_with_knn_attack:
        eval_metric = evaluate_with_knn_attack(model, eval_dataloader, tokenizer, metric, config, label_list, dataset_word_dict)
        if model_args.use_wandb:
            for key,value in eval_metric.items():
                wandb.log({f'metric/{key}':value})
    if model_args.train_inversion_model:
        model_attack_acc = train_inversion_model(config, tokenizer, model, train_dataloader, eval_dataloader, dataset_word_dict, model_args.use_wandb)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
