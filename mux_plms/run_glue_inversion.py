#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import evaluate
from datasets import load_dataset, load_metric
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from datamux_pretraining.models.multiplexing_legacy import (
    RobertaSequenceClassificationMuxed,
)
from datamux_pretraining.models.multiplexing_pretraining_bert import (
    MuxedBertForSequenceClassification,
)
from datamux_pretraining.models.multiplexing_pretraining_electra import (
    MuxedElectraForSequenceClassification,
)
from datamux_pretraining.models.finetune_trainer import FinetuneTrainer

from datamux_pretraining.models.tokenshuffle_data_collator import tokenshuffle_data_collator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from rouge_score import rouge_scorer
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import pickle
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "imdb": ("text", None),
    "ag_news":("text", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


version_2_modelcls = {
    "electra": MuxedElectraForSequenceClassification,
    "datamux_legacy": RobertaSequenceClassificationMuxed,
    "bert": MuxedBertForSequenceClassification,
}

class InversionPLM(nn.Module):
    def __init__(self, config, model_name_or_path='bert-base-uncased'):
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
    
class InversionPLMForPositionAttack(nn.Module):
    def __init__(self, config, model_name_or_path='bert-base-uncased'):
        super(InversionPLMForPositionAttack, self).__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, num_labels=config.sequence_length)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, label, attention_mask=None):
        outputs = self.model(inputs_embeds=x, labels=label, attention_mask=attention_mask)
        return outputs.logits, outputs.loss

    def predict(self, x, label=None, attention_mask=None):
        outputs = self.model(inputs_embeds=x, labels=label, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(F.softmax(logits,dim=-1), dim=2)
        return logits, pred

class InversionPLMMLC(nn.Module):
    def __init__(self, config, model_name_or_path='bert-base-uncased'):
        super(InversionPLMMLC,self).__init__()
        self.vocab_size = config.vocab_size
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

        self.loss = torch.nn.BCELoss()
        self.sigmod = torch.nn.Sigmoid()

    def forward(self, x, labels=None, attention_mask=None, token_type_ids=None):
        bsz, seq_len, hid_dim = x.shape
        device = x.device
    
        logits = self.model(inputs_embeds=x, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        
        logits = self.sigmod(torch.mean(logits, dim=1)) # (bsz, dim)

        loss = None
        if labels is not None:
            labels = torch.zeros(bsz, self.vocab_size).to(device).scatter_(1, labels, 1.)
            labels[:,0:3] = 0
            loss = self.loss(logits, labels)
        return logits, loss

    def predict(self, x, labels=None, attention_mask=None, token_type_ids=None):
        logits = self.model(inputs_embeds=x, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        pred = torch.round(self.sigmod(torch.mean(logits, dim=1))) # (bsz, vocab_size)
        return logits, pred

def mux_token_selection(model, filter_tokens, batch, real_sentence_idx, dataset_word_dict, select_strategy='None',  token2cluster=None, clusters2token_list=None, tokenizer=None): 
    batch_size, sequence_length = batch['input_ids'].size()
    emb = model.bert.embeddings.word_embeddings.weight
    dataset_word_dict = torch.tensor(dataset_word_dict)
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
        largest = True if select_strategy=='all_far' else False
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
    elif select_strategy == 'real_data':
        # 筛选出要替换的词
        invalid_ids = (batch['attention_mask'] == 0)
        invalid_ids[word_filter(batch['input_ids'], filter_tokens)] = True
        # 真实句子的词不进行替换操作
        invalid_ids[real_sentence_idx] = False
        # 只对假句子长度不足的情况进行替换操作
        real_sentence_length = list(batch['input_ids'][real_sentence_idx]).index(102)
        invalid_ids[:,real_sentence_length:] = False
        # 生成填充的batch
        filled_input_ids = torch.clone(batch['input_ids'])
        for sequence_idx in range(batch_size):
            sep_idx = list(batch['input_ids'][sequence_idx]).index(102)
            simple_token_ids = (filled_input_ids[sequence_idx]==-1)
            simple_token_ids[word_filter(filled_input_ids[sequence_idx], filter_tokens)] = True
            simple_token_ids[sep_idx:] = True
            sample_ids = (simple_token_ids!=True)
            sample_ids[0] = False
            sample_pool = batch['input_ids'][sequence_idx][sample_ids]
            # 生成待替换词的随机下标
            selection_ids = torch.randint(low=0, high=len(sample_pool)-1, size=filled_input_ids[sequence_idx][simple_token_ids].size())
            filled_input_ids[sequence_idx][simple_token_ids] = sample_pool[selection_ids]
        # 把假句子用自身填满
        filled_input_ids[real_sentence_idx] = batch['input_ids'][real_sentence_idx]
        batch['input_ids'][invalid_ids] = filled_input_ids[invalid_ids]
        batch['input_ids'][:,real_sentence_length-1] = 1012
        batch['input_ids'][:,real_sentence_length:] = 0
        batch['input_ids'][:,real_sentence_length] = 102
    elif select_strategy=="cluster":
        real_sentence_length = list(batch['input_ids'][real_sentence_idx]).index(102)
        real_sentence = batch['input_ids'][real_sentence_idx]
        for idx in range(1, real_sentence_length):
            cur_token = real_sentence[idx]
            # cluster sample
            cluster_id = token2cluster[int(cur_token.item())]
            cluster_sample_pool = clusters2token_list[cluster_id]
            cluster_sample_pool_repeat = (batch_size-1) // len(cluster_sample_pool) + 1
            if cluster_sample_pool_repeat > 1 :
                cluster_sample_pool = [token for token in cluster_sample_pool for repeat_times in range(cluster_sample_pool_repeat)]
            cluster_selected_tokens = random.sample(cluster_sample_pool, k=batch_size-1)
            selected_tokens = cluster_selected_tokens
            selected_tokens.insert(real_sentence_idx, cur_token)
            selected_tokens = torch.tensor(selected_tokens)
            batch['input_ids'][:,idx] = selected_tokens
    elif select_strategy=="input_self":
        real_sentence_length = list(batch['input_ids'][real_sentence_idx]).index(102)
        real_sentence = batch['input_ids'][real_sentence_idx]
        real_sentence_sample_num = batch_size-1 # [0, num_instances-1] or (batch_size-1) // 2
        sample_pool = list(real_sentence[1:real_sentence_length])
        # # 去掉采样池中的简单词
        # for token in sample_pool:
        #     if token in filter_tokens:
        #         sample_pool.remove(token)
        if len(sample_pool) <= 1 :
            selection_ids = torch.randint(low=0, high=len(dataset_word_dict)-1, size=(real_sentence_sample_num,))
            selection_tokens = dataset_word_dict[selection_ids].to('cuda')
            sample_pool = list(selection_tokens)
        real_sentence_sample_pool_repeat = (real_sentence_sample_num // (len(sample_pool)-1)) + 1
        for idx in range(1, real_sentence_length):
            cur_token = real_sentence[idx]
            # real sentence sample
            real_sentence_sample_pool = sample_pool[:]
            if cur_token in real_sentence_sample_pool:
                real_sentence_sample_pool.remove(cur_token)
            real_sentence_sample_pool *= real_sentence_sample_pool_repeat
            real_sentence_selected_tokens = random.sample(real_sentence_sample_pool, k=real_sentence_sample_num)
            selected_tokens = real_sentence_selected_tokens
            selected_tokens.insert(real_sentence_idx, cur_token)
            selected_tokens = torch.tensor(selected_tokens)
            batch['input_ids'][:,idx] = selected_tokens
        batch['input_ids'][:,real_sentence_length-1] = 1012
        batch['input_ids'][:,real_sentence_length:] = 0
        batch['input_ids'][:,real_sentence_length] = 102
    elif select_strategy=="cluster_realsen":
        real_sentence_length = list(batch['input_ids'][real_sentence_idx]).index(102)
        real_sentence = batch['input_ids'][real_sentence_idx]
        real_sentence_sample_num = 6 # [0, num_instances-1] or (batch_size-1) // 2
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
    
def dataloader2memory(dataloader, model, tokenizer, num_instances, dataset_word_dict, select_strategy, task_name, target_layer=3, device='cuda'):
    token2cluster = None
    clusters2token_list = None
    features = []
    pro_bar = tqdm(range(len(dataloader)))
    model.eval()
    # filter special tokens
    special_tokens = tokenizer.convert_tokens_to_ids(['[PAD]','[SEP]'])
    simple_tokens = tokenizer.convert_tokens_to_ids(['.', ',', '"', '-',"'",'(',')',':',';','`','<','>','#','the','a','t','n','?','%','/','\\','&','$','of','br','and','s','##s','to','is','was','for','that','in','as','on'])
    filter_tokens = list(set(special_tokens + simple_tokens))
    if "cluster" in select_strategy:
        dataset_word_num = len(dataset_word_dict)
        n_clusters = dataset_word_num // (num_instances * 10)
        # save result
        cluster_dir = f'kmeans_cluster_num{n_clusters}'
        cluster_path = os.path.join(f'/root/mixup/mux_plms/cluster/{task_name}', cluster_dir)
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
            all_position_attack_label = []
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
                if select_strategy != 'None':
                    mux_minibatch = mux_token_selection(model, filter_tokens, mux_minibatch, real_sentence_idx, dataset_word_dict, select_strategy, token2cluster, clusters2token_list, tokenizer)
                mux_minibatch['real_sentence_idx'] = real_sentence_idx
                outputs = model(**mux_minibatch)
                hidden_states = outputs.hidden_states
                all_mux_sentence_input_ids.append(mux_minibatch['input_ids'])
                all_hidden_states.append(hidden_states)
                all_position_attack_label.append(outputs.position_attack_label)
            input_ids = batch['input_ids'].to('cpu')
            attention_mask = batch['attention_mask'].to('cpu')
            target_hidden_states = torch.cat(all_hidden_states, dim=0).to('cpu')
            all_mux_sentence_input_ids = torch.stack(all_mux_sentence_input_ids)
            all_position_attack_label = torch.stack(all_position_attack_label)
            features.append({'hidden_states': target_hidden_states, 'input_ids': input_ids, 'attention_mask': attention_mask, 'mux_sentence_input_ids': all_mux_sentence_input_ids, "position_attack_label": all_position_attack_label})
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

def train_inversion_model(config, tokenizer, model, train_dataloader, eval_dataloader, dataset_word_dict, use_wandb=True, output_dir=None):
    # batch_size=32 # {roberta:32, mlp:64}
    learning_rate=2e-5 # {roberta:1e-5 2e-5 5e-5, mlp:2e-4}
    device='cuda'
    epochs=20
    topk = 1
    inversion_model = InversionPLM(config)

    inversion_model = inversion_model.to(device)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(inversion_model.parameters(), lr=learning_rate)

    print('load dataloader to memory')
    train_dataloader = dataloader2memory(train_dataloader, model, tokenizer, config.num_instances, dataset_word_dict, config.select_strategy, config.task_name, config.target_layer, device)
    eval_dataloader = dataloader2memory(eval_dataloader, model, tokenizer, config.num_instances, dataset_word_dict, config.select_strategy, config.task_name, config.target_layer, device)
    print('done')
    
    total_step = len(train_dataloader) * epochs
    
    progress_bar = tqdm(range(total_step))
    special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    # filted inversion
    simple_tokens = tokenizer.convert_tokens_to_ids(['.', ',', '"', '-',"'",'(',')',':',';','`','<','>','#','the','a','t','n','?','%','/','\\','&','$','of','br','and','s','##s','to','is','was','for','that','in','as','on'])
    # origin inversion
    # simple_tokens = tokenizer.convert_tokens_to_ids(['.', ',', '"', '-'])
    filter_tokens = list(set(special_tokens + simple_tokens))
    completed_steps = 0
    model_attack_acc = 0
    hit_tokens = {}
    mux_tokens_list = []
    # best
    best_top1_acc = 0
    best_top5_acc = 0
    best_rouge = 0
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
            
        if (epoch+1) %4 == 0:
            with torch.no_grad():
                # hit
                rouge_hit_cnt = 0
                top1_hit_cnt = 0
                top5_hit_cnt = 0
                # total
                total_cnt = 0
                rouge_total_cnt = 0
                for batch in eval_dataloader:
                    batch = {key:value.to(device) for key,value in batch.items()}
                    target_hidden_states = batch['hidden_states']
                    
                    eval_label = batch['input_ids']
                    attention_mask = batch['attention_mask']

                    feature = target_hidden_states
                    feature = feature.to(device)
                    attention_mask = attention_mask.to(device)
                    pred_logits, preds = inversion_model.predict(feature, attention_mask=attention_mask)

                    valid_ids = attention_mask!=0
                    valid_ids[word_filter(eval_label, filter_tokens)] = False
                    eval_label = batch['input_ids']
                    eval_label = eval_label[valid_ids] 
                    # inversion top1
                    top1_preds = torch.topk(pred_logits, k=1)[1]
                    top1_preds = top1_preds[valid_ids]
                    top1_hit_cnt += (eval_label.unsqueeze(1) == top1_preds).int().sum().item()
                    # inversion top5
                    top5_preds = torch.topk(pred_logits, k=5)[1]
                    top5_preds = top5_preds[valid_ids]
                    top5_hit_cnt += (eval_label.unsqueeze(1) == top5_preds).int().sum().item()
                    total_cnt += eval_label.shape[0]
                    # rouge
                    r_hit_cnt, r_total_cnt = rouge(eval_label.unsqueeze(1), top1_preds, tokenizer)
                    rouge_hit_cnt += r_hit_cnt
                    rouge_total_cnt += r_total_cnt
                # caculate attack accuracy
                top1_model_attack_acc = top1_hit_cnt/total_cnt
                top5_model_attack_acc = top5_hit_cnt/total_cnt
                rouge_acc = rouge_hit_cnt / rouge_total_cnt
                print('eval inversion top1 attack acc:{}'.format(top1_model_attack_acc))
                if use_wandb:
                    wandb.log({'eval/inversion_model_top1_acc': top1_model_attack_acc})
                    wandb.log({'eval/inversion_model_top5_acc': top5_model_attack_acc})
                    wandb.log({'eval/inversion_model_rouge_acc': rouge_acc})
                # record the best
                if top1_model_attack_acc > best_top1_acc:
                    best_top1_acc = top1_model_attack_acc
                if top5_model_attack_acc > best_top5_acc:
                    best_top5_acc = top5_model_attack_acc
                if rouge_acc > best_rouge:
                    best_rouge = rouge_acc

        if epoch == epochs - 1:
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
                # model_attack_acc = hit_cnt/total_cnt
                # print('attack acc:{}'.format(hit_cnt/total_cnt))
                # if use_wandb:
                #     wandb.log({'metric/inversion_model_top{}_acc'.format(topk): hit_cnt/total_cnt})
    
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
    # log the best
    print(f'best_inversion_model_top1_acc:{best_top1_acc}')
    print(f'best_inversion_model_top5_acc:{best_top5_acc}')
    print(f'best_inversion_model_rouge:{best_rouge}')
    if use_wandb:
        wandb.log({'best/best_inversion_model_top1_acc': best_top1_acc})
        wandb.log({'best/best_inversion_model_top5_acc': best_top5_acc})
        wandb.log({'best/best_inversion_model_rouge_acc': best_rouge})
    # save inversion model
    torch.save(inversion_model, os.path.join(output_dir,'inversion_model.pt'))
    
    # train mlc model
    train_mlc_model(config, tokenizer, model, train_dataloader, eval_dataloader, dataset_word_dict, use_wandb=use_wandb, output_dir=output_dir, inversion_epochs=20, inversion_lr=5e-5)
    return model_attack_acc

def train_position_inversion_model(config, tokenizer, model, train_dataloader, eval_dataloader, dataset_word_dict, use_wandb=True, output_dir=None):
    # batch_size=32 # {roberta:32, mlp:64}
    learning_rate=2e-5 # {roberta:1e-5 2e-5 5e-5, mlp:2e-4}
    device='cuda'
    epochs=20
    inversion_model = InversionPLMForPositionAttack(config)

    inversion_model = inversion_model.to(device)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(inversion_model.parameters(), lr=learning_rate)

    print('load dataloader to memory')
    train_dataloader = dataloader2memory(train_dataloader, model, tokenizer, config.num_instances, dataset_word_dict, config.select_strategy, config.task_name, config.target_layer, device)
    eval_dataloader = dataloader2memory(eval_dataloader, model, tokenizer, config.num_instances, dataset_word_dict, config.select_strategy, config.task_name, config.target_layer, device)
    print('done')
    
    total_step = len(train_dataloader) * epochs
    
    progress_bar = tqdm(range(total_step))
    special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    filter_tokens = list(set(special_tokens))
    completed_steps = 0
    # best
    best_top1_acc = 0
    print('################# start train inversion model #################')
    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            batch = {key:value.to(device) for key,value in batch.items()}
            target_hidden_states = batch['hidden_states']
            labels = batch['position_attack_label']
            
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
            
        if (epoch+1) %4 == 0:
            with torch.no_grad():
                # hit
                top1_hit_cnt = 0
                # total
                total_cnt = 0
                for batch in eval_dataloader:
                    batch = {key:value.to(device) for key,value in batch.items()}
                    target_hidden_states = batch['hidden_states']
                    
                    eval_label = batch['position_attack_label']
                    attention_mask = batch['attention_mask']

                    feature = target_hidden_states
                    feature = feature.to(device)
                    attention_mask = attention_mask.to(device)
                    pred_logits, preds = inversion_model.predict(feature, attention_mask=attention_mask)

                    valid_ids = attention_mask!=0
                    valid_ids[word_filter(batch['input_ids'], filter_tokens)] = False
                    eval_label = eval_label[valid_ids]
                    # inversion top1
                    top1_preds = torch.topk(pred_logits, k=1)[1]
                    top1_preds = top1_preds[valid_ids]
                    top1_hit_cnt += (eval_label.unsqueeze(1) == top1_preds).int().sum().item()
                    total_cnt += eval_label.shape[0]
                # caculate attack accuracy
                top1_model_attack_acc = top1_hit_cnt/total_cnt
                print('eval position inversion ttack acc:{}'.format(top1_model_attack_acc))
                if use_wandb:
                    wandb.log({'eval/position_inversion_model_top1_acc': top1_model_attack_acc})
                # record the best
                if top1_model_attack_acc > best_top1_acc:
                    best_top1_acc = top1_model_attack_acc
    # log the best
    print(f'best_position_inversion_model_top1_acc:{best_top1_acc}')
    if use_wandb:
        wandb.log({'best/best_position_inversion_model_top1_acc': best_top1_acc})


def token_hit(input_ids, pred_ids, tokenizer, filter_tokens):
    batch_real_tokens = [tokenizer.convert_ids_to_tokens(item) for item in input_ids]
    batch_pred_tokens = [tokenizer.convert_ids_to_tokens(item) for item in pred_ids]
    hit_cnt = 0
    total_cnt = 0
    for real_tokens, pred_tokens in zip(batch_real_tokens, batch_pred_tokens):
        real_tokens = {item for item in set(real_tokens) if item not in filter_tokens}
        pred_tokens = {item for item in set(pred_tokens) if item not in filter_tokens}
        hit_cnt += len(real_tokens & pred_tokens)
        total_cnt += len(real_tokens)
    return hit_cnt, total_cnt

def train_mlc_model(config, tokenizer, model, train_dataloader, eval_dataloader, dataset_word_dict, use_wandb=True, output_dir=None, inversion_epochs=20, inversion_lr=5e-5):
    device ='cuda'
    learning_rate = inversion_lr # {roberta:5e-5, mlp:2e-4}
    epochs = inversion_epochs
    inversion_model = InversionPLMMLC(config)

    inversion_model = inversion_model.to(device)
    model = model.to(device)
    
    # print('load dataloader to memory')
    # train_dataloader = dataloader2memory(train_dataloader, model, tokenizer, config.num_instances, dataset_word_dict, config.select_strategy, config.task_name, config.target_layer, device)
    # eval_dataloader = dataloader2memory(eval_dataloader, model, tokenizer, config.num_instances, dataset_word_dict, config.select_strategy, config.task_name, config.target_layer, device)
    # print('done')
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in inversion_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in inversion_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    total_step = len(train_dataloader) * epochs

    progress_bar = tqdm(range(total_step))
    special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    # filted inversion
    simple_tokens = tokenizer.convert_tokens_to_ids(['.', ',', '"', '-',"'",'(',')',':',';','`','<','>','#','the','a','t','n','?','%','/','\\','&','$','of','br','and','s','##s','to','is','was','for','that','in','as','on'])
    # origin inversion
    # simple_tokens = tokenizer.convert_tokens_to_ids(['.', ',', '"', '-'])
    filter_tokens = list(set(special_tokens + simple_tokens))
    
    completed_steps = 0
    model_attack_acc = 0
    best_eval_attack_acc = 0
    print('################# start train mlc model #################')
    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            batch = {key:value.to(device) for key,value in batch.items()}
    
            target_hidden_states = batch['hidden_states']
            labels = batch['input_ids']
            attention_mask = batch['attention_mask']
            attention_mask[word_filter(labels, filter_tokens)] = 0 
            
            logits, loss = inversion_model(target_hidden_states, labels, attention_mask=attention_mask)
            
            if use_wandb:
                wandb.log({'loss/inversion_model_loss':loss.item()})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            completed_steps += 1
            progress_bar.update(1)
            progress_bar.set_description('mlc_model_loss:{}'.format(loss.item()))

        if True:
            hit_cnt = 0
            total_cnt = 0
            for batch in eval_dataloader:
                batch = {key:value.to(device) for key,value in batch.items()}
                
                target_hidden_states = batch['hidden_states']
                eval_label = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                pred_logits, batch_preds = inversion_model.predict(target_hidden_states, attention_mask=attention_mask)
                bsz, _ = batch_preds.shape
                batch_eval_label = batch['input_ids']
                for i in range(bsz):
                    preds = batch_preds[i].nonzero().squeeze(-1).unsqueeze(0)
                    eval_label = batch_eval_label[i].unsqueeze(0)
                    temp_hit, temp_total = token_hit(eval_label, preds, tokenizer, filter_tokens)
                    hit_cnt += temp_hit
                    total_cnt += temp_total
            print('eval mlc attack acc:{}'.format(hit_cnt/total_cnt))
            if use_wandb:
                wandb.log({'metric/eval_mlc_model_acc': hit_cnt/total_cnt})
            if hit_cnt/total_cnt > best_eval_attack_acc:
                best_eval_attack_acc = hit_cnt/total_cnt
    print(f'best_eval_mlc_model_acc:{best_eval_attack_acc}')
    if use_wandb:
        wandb.log({'best/best_mlc_model_acc': best_eval_attack_acc})
     # save inversion model
    torch.save(inversion_model, os.path.join(output_dir,'mlc_model.pt'))


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

def evaluate_with_knn_attack(model, dataloader, tokenizer, metric, config, dataset_word_dict):
    emb = model.bert.embeddings.word_embeddings.weight
    device = 'cuda'
    select_strategy = config.select_strategy
    token2cluster = None
    clusters2token_list = None
    if "cluster" in select_strategy:
        dataset_word_num = len(dataset_word_dict)
        n_clusters = dataset_word_num // (config.num_instances * 10)
        # save result
        cluster_dir = f'kmeans_cluster_num{n_clusters}'
        cluster_path = os.path.join(f'/root/mixup/mux_plms/cluster/{config.task_name}', cluster_dir)
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
                if select_strategy != 'None':
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
        references = batch["labels"]
        metric.add_batch(
            predictions=all_predictions,
            references=references,
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
    eval_metric = metric.compute()
    eval_metric['knn_top1'] = top1_hit_cnt/total_cnt
    eval_metric['knn_top5'] = top5_hit_cnt/total_cnt
    eval_metric['knn_top10'] = top10_hit_cnt/total_cnt
    eval_metric['knn_rouge'] = rouge_hit_cnt/rouge_total_cnt
    return eval_metric

def cluster_pipeline(model, dataset_word_dict, task_name, num_instances):
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
    cluster_path = os.path.join(f'/root/mixup/mux_plms/cluster/{task_name}', cluster_dir)
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
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys.keys())
        },
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
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
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

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in "
                    + ",".join(task_to_keys.keys())
                )
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                "Need either a GLUE task, a training/validation file or a dataset name."
            )
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


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
        default=0,
        metadata={"help": "Coefficient for retrieval loss"},
    )
    task_loss_coeff: Optional[float] = field(
        default=1.0,
        metadata={"help": "Coefficient for task loss"},
    )
    epsilon: Optional[float] = field(
        default=1.0,
        metadata={"help": "the scale for laplace noise"},
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
    position_encryption: bool = field(
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
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
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

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        if data_args.task_name == "imdb":
            datasets = load_dataset("imdb")
        elif data_args.task_name == "ag_news":
            datasets = load_dataset("ag_news")
        else:
            datasets = load_dataset(
                "glue", data_args.task_name, cache_dir=model_args.cache_dir
            )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {
            "train": data_args.train_file,
            "validation": data_args.validation_file,
        }

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError(
                    "Need either a GLUE task or a test file for `do_predict`."
                )

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset(
                "csv", data_files=data_files, cache_dir=model_args.cache_dir
            )
        else:
            # Loading a dataset from local json files
            datasets = load_dataset(
                "json", data_files=data_files, cache_dir=model_args.cache_dir
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in [
            "float32",
            "float64",
        ]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    #
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
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
            use_fast=model_args.use_fast_tokenizer,
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
    config.add_noise = 0
    config.token_shuffle = 0
    config.add_embedding_noise = model_args.add_embedding_noise
    config.epsilon = model_args.epsilon
    config.target_layer = 3
    config.wandb_name = model_args.wandb_name
    config.task_name = data_args.task_name
    config.select_strategy = model_args.select_strategy
    config.position_encryption = model_args.position_encryption
    config.sequence_length = data_args.max_seq_length
    
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
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
            )
        else:
            model = AutoModelForSequenceClassification.from_config(config)

    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in datasets["train"].column_names if name != "label"
        ]
        if (
            "sentence1" in non_label_column_names
            and "sentence2" in non_label_column_names
        ):
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )
        
        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [
                (label_to_id[l] if l != -1 else -1) for l in examples["label"]
            ]
        return result

    datasets = datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=False,
        num_proc=data_args.preprocessing_num_workers,
    )
    # if training_args.do_train:
    if "train" not in datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = datasets["train"]
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))

    # if training_args.do_eval:
    # if "validation" not in datasets and "validation_matched" not in datasets:
    #     raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = datasets[
        "test" if data_args.task_name == "imdb" or data_args.task_name == "ag_news" else "validation"
    ]
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if (
        training_args.do_predict
        or data_args.task_name is not None
        or data_args.test_file is not None
    ):
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = datasets[
            "test_matched" if data_args.task_name == "mnli" else "test"
        ]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(
                range(data_args.max_predict_samples)
            )

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        if data_args.task_name == "imdb" or data_args.task_name == "ag_news":
            metric = evaluate.load("accuracy")
        else:
            metric = evaluate.load("glue", data_args.task_name)
    else:
        metric = evaluate.load("accuracy")
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
        
    if config.token_shuffle:
        data_collator = tokenshuffle_data_collator

    if model_args.use_wandb:
        project_name = f'muxplm_{data_args.task_name}_mux{config.num_instances}'
        wandb.init(config=config, project=project_name, entity='privacy_cluster', name=model_args.wandb_name, sync_tensorboard=False,
                job_type="CleanRepo", settings=wandb.Settings(start_method='fork'))
        
    trainer = FinetuneTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # Training
    if training_args.do_train:
        logger.info("*** Train ***")

        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        output_dir = training_args.output_dir
        last_checkpoint = get_last_checkpoint(output_dir)
        logger.info("last checkpoint: %s", last_checkpoint)

        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        tasks = [data_args.task_name]
        all_metrics = {}
        for seed in range(1):
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

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset.remove_columns_("label")
            predictions = trainer.predict(
                predict_dataset, metric_key_prefix="predict"
            ).predictions
            predictions = (
                np.squeeze(predictions)
                if is_regression
                else np.argmax(predictions, axis=1)
            )

            output_predict_file = os.path.join(
                training_args.output_dir, f"predict_results_{task}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=60, drop_last=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=60, drop_last=True)
    
    torch.cuda.empty_cache()
    print('statistic dataset word dict')
    dataset_word_dict = get_dataset_word_dict(train_dataset, eval_dataset)
    print('done')
    # if cluster have not been done, do cluster
    if 'cluster' in config.select_strategy:
        dataset_word_num = len(dataset_word_dict)
        cluster_num = dataset_word_num // (config.num_instances * 10)
        cluster_dir = f'kmeans_cluster_num{cluster_num}'
        cluster_path = os.path.join(f'/root/mixup/mux_plms/cluster/{config.task_name}', cluster_dir)
        if not os.path.exists(cluster_path):
            cluster_pipeline(model, dataset_word_dict, config.task_name, config.num_instances)
    # set_seed(3)
    if model_args.eval_with_knn_attack:
        eval_metric = evaluate_with_knn_attack(model, eval_dataloader, tokenizer, metric, config, dataset_word_dict)
        if model_args.use_wandb:
            for key,value in eval_metric.items():
                wandb.log({f'metric/{key}':value})
    if model_args.train_inversion_model:
        # train_position_inversion_model(config, tokenizer, model, train_dataloader, eval_dataloader, dataset_word_dict, model_args.use_wandb, training_args.output_dir)
        model_attack_acc = train_inversion_model(config, tokenizer, model, train_dataloader, eval_dataloader, dataset_word_dict, model_args.use_wandb, training_args.output_dir)
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
