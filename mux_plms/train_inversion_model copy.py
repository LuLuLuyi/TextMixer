# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning a 🤗🤗🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from torch.distributions.laplace import Laplace
import wandb

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
    
    
def evaluate_with_knn_attack(model, dataloader, metric, accelerator, topk=5, target_layer=3):
    model.eval()
    samples_seen = 0
    token_shuffle = True
    for step, batch in enumerate(dataloader):
        # batch['output_hidden_states'] = True
        batch_size, sequence_length = batch['input_ids'].size()
        # token shuffle
        if token_shuffle:
            for idx in range(sequence_length):
                shuffled_idx = random.sample(range(0,10), 10)
                batch['input_ids'][:, idx] = batch['input_ids'][shuffled_idx, idx]
        with torch.no_grad():
            outputs = model(**batch)
       
        # evaluate
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(dataloader) - 1:
                predictions = predictions[: len(dataloader.dataset) - samples_seen]
                references = references[: len(dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
        
    eval_metric = metric.compute()
    return eval_metric
    
def dataloader2memory(dataloader, model, target_layer=3, device='cuda'):
    token_shuffle = True
    features = []
    pro_bar = tqdm(range(len(dataloader)))
    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            batch = {key:value.to(device) for key,value in batch.items()}
            batch_size, sequence_length = batch["input_ids"].size()
            all_hidden_states = [] 
            all_mux_sentence_ids = []
            for idx in range(batch_size):
                sample_list = list(range(0,batch_size))
                sample_list.remove(idx)
                mux_sentence_ids = random.sample(sample_list, k=10-1)
                mux_sentence_ids.insert(random.randint(0, len(mux_sentence_ids)),idx)
                
                mux_minibatch = {key:value[mux_sentence_ids] for key,value in batch.items()} 
                # token shuffle
                # if token_shuffle:
                #     for idx in range(sequence_length):
                #         shuffled_idx = random.sample(range(0,10), 10)
                #         mux_minibatch['input_ids'][:, idx] = mux_minibatch['input_ids'][shuffled_idx, idx]
                outputs = model(**mux_minibatch)
                hidden_states = outputs.hidden_states[target_layer][:,11:,]
                
                all_mux_sentence_ids.append(mux_sentence_ids)
                all_hidden_states.append(hidden_states)
            input_ids = batch['input_ids'].to('cpu')
            attention_mask = batch['attention_mask'].to('cpu')
            target_hidden_states = torch.cat(all_hidden_states, dim=0).to('cpu')
            all_mux_sentence_ids = torch.tensor(all_mux_sentence_ids)
            features.append({'hidden_states': target_hidden_states, 'input_ids': input_ids, 'attention_mask': attention_mask, 'mux_sentence_ids': all_mux_sentence_ids})
        pro_bar.update(1)
    return features

def word_filter(eval_label, filter_list):
    allow_token_ids = (eval_label == filter_list[0])
    for item in filter_list:
        allow_token_ids = allow_token_ids | (eval_label == item)
    return allow_token_ids

# def train_inversion_model(config, tokenizer, model, train_dataloader, eval_dataloader, metric, accelerator, use_wandb=True):
#     batch_size=32 # {roberta:32, mlp:64}
#     learning_rate=5e-5 # {roberta:5e-5, mlp:2e-4}
#     device='cuda'
#     epochs=20
#     topk = 1
#     inversion_model = InversionPLM(config)

#     inversion_model = inversion_model.to(device)
#     model = model.to(device)
    
#     optimizer = torch.optim.AdamW(inversion_model.parameters(), lr=learning_rate)
    
#     print('load dataloader to memory')
#     train_dataloader = dataloader2memory(train_dataloader, model, config.target_layer, device)
#     eval_dataloader = dataloader2memory(eval_dataloader, model, config.target_layer, device)
#     print('done')
    
#     knn_attack(model, eval_dataloader, use_wandb, target_layer=config.target_layer, topk=10)
    
#     total_step = len(train_dataloader) * epochs
#     lr_scheduler = get_scheduler(
#         name='linear',
#         optimizer=optimizer,
#         num_warmup_steps=0,
#         num_training_steps=total_step,
#     )
    
#     progress_bar = tqdm(range(total_step))
#     special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
#     simple_tokens = tokenizer.convert_tokens_to_ids(['.', ',', '"', '-'])
#     filter_tokens = list(set(special_tokens + simple_tokens))
    
#     completed_steps = 0
#     model_attack_acc = 0
#     print('################# start train inversion model #################')
#     for epoch in range(epochs):
#         for step, batch in enumerate(train_dataloader):
#             batch = {key:value.to(device) for key,value in batch.items()}
#             target_hidden_states = batch['hidden_states']
#             labels = batch['input_ids']
#             labels[labels == tokenizer.pad_token_id] = -100
            
#             attention_mask = batch['attention_mask']
            
#             bsz, seq_len, dim = target_hidden_states.shape
#             feature = target_hidden_states
            
#             feature = feature.to(device)
#             attention_mask = attention_mask.to(device)
#             labels = labels.to(device)
            
#             logits, loss = inversion_model(feature, labels, attention_mask=attention_mask)
#             if use_wandb:
#                 wandb.log({'loss/inversion_model_loss':loss.item()})

#             loss.backward()
#             optimizer.step()
#             # lr_scheduler.step()
#             optimizer.zero_grad()
#             completed_steps += 1
#             progress_bar.update(1)
#             progress_bar.set_description('inversion_model_loss:{}'.format(loss.item()))

#         if True:
#             hit_cnt = 0
#             total_cnt = 0
#             for batch in eval_dataloader:
#                 batch = {key:value.to(device) for key,value in batch.items()}
#                 # batch['output_hidden_states'] = True
                
#                 # outputs = model(**batch)
#                 # target_hidden_states = outputs.hidden_states[target_layer]
#                 target_hidden_states = batch['hidden_states']
#                 eval_label = batch['input_ids']
#                 attention_mask = batch['attention_mask']

#                 bsz, seq_len, dim = target_hidden_states.shape
#                 feature = target_hidden_states
#                 feature = feature.to(device)
#                 attention_mask = attention_mask.to(device)

#                 # feature = torch.cat([feature[:, 0], feature[:, 1]], dim=2)
#                 pred_logits, preds = inversion_model.predict(feature, attention_mask=attention_mask)

#                 valid_ids = attention_mask!=0
                
                
#                 valid_ids[word_filter(eval_label, filter_tokens)] = False
#                 eval_label = batch['input_ids']
#                 eval_label = eval_label[valid_ids] 
#                 preds = torch.topk(pred_logits, k=topk)[1]
#                 preds = preds[valid_ids]
#                 hit_cnt += (eval_label.unsqueeze(1) == preds).int().sum().item()
#                 total_cnt += eval_label.shape[0]
#             model_attack_acc = hit_cnt/total_cnt
#             print('attack acc:{}'.format(hit_cnt/total_cnt))
#             if use_wandb:
#                 wandb.log({'metric/inversion_model_top{}_acc'.format(topk): hit_cnt/total_cnt})
#     return model_attack_acc

# def train_inversion_model(config, tokenizer, model, train_dataloader, eval_dataloader, use_wandb=True):
#     batch_size=32 # {roberta:32, mlp:64}
#     learning_rate=5e-5 # {roberta:5e-5, mlp:2e-4}
#     device='cuda'
#     epochs=20
#     topk = 1
#     inversion_model = InversionPLM(config)

#     inversion_model = inversion_model.to(device)
#     model = model.to(device)
    
#     optimizer = torch.optim.AdamW(inversion_model.parameters(), lr=learning_rate)
    
#     print('load dataloader to memory')
#     train_dataloader = dataloader2memory(train_dataloader, model, config.target_layer, device)
#     eval_dataloader = dataloader2memory(eval_dataloader, model, config.target_layer, device)
#     print('done')
    
#     total_step = len(train_dataloader) * epochs
#     lr_scheduler = get_scheduler(
#         name='linear',
#         optimizer=optimizer,
#         num_warmup_steps=0,
#         num_training_steps=total_step,
#     )
    
#     progress_bar = tqdm(range(total_step))
#     special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
#     simple_tokens = tokenizer.convert_tokens_to_ids(['.', ',', '"', '-'])
#     filter_tokens = list(set(special_tokens + simple_tokens))
    
#     completed_steps = 0
#     model_attack_acc = 0
#     print('################# start train inversion model #################')
#     for epoch in range(epochs):
#         for step, batch in enumerate(train_dataloader):
#             batch = {key:value.to(device) for key,value in batch.items()}
#             target_hidden_states = batch['hidden_states']
#             labels = batch['input_ids']
#             labels[labels == tokenizer.pad_token_id] = -100
            
#             attention_mask = batch['attention_mask']
            
#             bsz, seq_len, dim = target_hidden_states.shape
#             feature = target_hidden_states
            
#             feature = feature.to(device)
#             attention_mask = attention_mask.to(device)
#             labels = labels.to(device)
            
#             logits, loss = inversion_model(feature, labels, attention_mask=attention_mask)
#             if use_wandb:
#                 wandb.log({'loss/inversion_model_loss':loss.item()})

#             loss.backward()
#             optimizer.step()
#             # lr_scheduler.step()
#             optimizer.zero_grad()
#             completed_steps += 1
#             progress_bar.update(1)
#             progress_bar.set_description('inversion_model_loss:{}'.format(loss.item()))

#         if True:
#             hit_cnt = 0
#             total_cnt = 0
#             with open(f'/root/mixup/mux_plms/case_study/datamux-sst2-10/inversion_epoch{epoch}.txt','w') as f:
#                 for batch in eval_dataloader:
#                     batch = {key:value.to(device) for key,value in batch.items()}
#                     # batch['output_hidden_states'] = True
                    
#                     # outputs = model(**batch)
#                     # target_hidden_states = outputs.hidden_states[target_layer]
#                     target_hidden_states = batch['hidden_states']
#                     eval_label = batch['input_ids']
#                     attention_mask = batch['attention_mask']

#                     bsz, seq_len, dim = target_hidden_states.shape
#                     feature = target_hidden_states
#                     feature = feature.to(device)
#                     attention_mask = attention_mask.to(device)

#                     # feature = torch.cat([feature[:, 0], feature[:, 1]], dim=2)
#                     pred_logits, preds = inversion_model.predict(feature, attention_mask=attention_mask)

#                     valid_ids = attention_mask!=0
                    
                    
#                     valid_ids[word_filter(eval_label, filter_tokens)] = False
#                     eval_label = batch['input_ids']
#                     eval_label = eval_label[valid_ids] 
#                     preds = torch.topk(pred_logits, k=topk)[1]
#                     preds = preds[valid_ids]
#                     hit_cnt += (eval_label.unsqueeze(1) == preds).int().sum().item()
#                     total_cnt += eval_label.shape[0]
#                     # 进行case_study记录
#                     top10_preds = torch.topk(pred_logits, k=10)[1]
#                     for idx in range(0, batch["input_ids"].size()[0], 2):
#                         f.write('-----------------------------mixup case start-----------------------------\n')
#                         f.write(f"被攻击的句子: {tokenizer.decode(batch['input_ids'][idx])} \n")
#                         f.write('参与mixup的句子:\n')
#                         f.write(f"sequence0 mixup weight:{batch['weight_matrix'][0][0]} context:{tokenizer.decode(batch['input_ids'][idx])} \n")
#                         f.write(f"sequence1 mixup weight:{batch['weight_matrix'][0][1]} context:{tokenizer.decode(batch['input_ids'][idx+1])} \n")
#                         f.write('inversion_top10_attack:\n')
#                         for j in range(10):
#                             f.write(f"top{j}: {tokenizer.decode(top10_preds[idx,:,j])} \n")
#                         f.write('\n')
#                         f.write(f"被攻击的句子: {tokenizer.decode(batch['input_ids'][idx+1])} \n")
#                         f.write('参与mixup的句子:\n')
#                         f.write(f"sequence0 mixup weight:{batch['weight_matrix'][1][0]} context:{tokenizer.decode(batch['input_ids'][idx])} \n")
#                         f.write(f"sequence1 mixup weight:{batch['weight_matrix'][1][1]} context:{tokenizer.decode(batch['input_ids'][idx+1])} \n")
#                         f.write('inversion_top10_attack:\n')
#                         for j in range(10):
#                             f.write(f"top{j}: {tokenizer.decode(top10_preds[idx+1,:,j])} \n")
#                         f.write('-----------------------------mixup case end-----------------------------\n')
#                         f.write('\n')

#             model_attack_acc = hit_cnt/total_cnt
#             print('attack acc:{}'.format(hit_cnt/total_cnt))
#             if use_wandb:
#                 wandb.log({'metric/inversion_model_top{}_acc'.format(topk): hit_cnt/total_cnt})
#     return model_attack_acc

def train_inversion_model(config, tokenizer, model, train_dataloader, eval_dataloader, use_wandb=True):
    batch_size=32 # {roberta:32, mlp:64}
    learning_rate=5e-5 # {roberta:5e-5, mlp:2e-4}
    device='cuda'
    epochs=20
    topk = 1
    inversion_model = InversionPLM(config)

    inversion_model = inversion_model.to(device)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(inversion_model.parameters(), lr=learning_rate)
    
    print('load dataloader to memory')
    train_dataloader = dataloader2memory(train_dataloader, model, config.target_layer, device)
    eval_dataloader = dataloader2memory(eval_dataloader, model, config.target_layer, device)
    print('done')
    
    total_step = len(train_dataloader) * epochs
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_step,
    )
    
    progress_bar = tqdm(range(total_step))
    special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
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

        if True:
            hit_cnt = 0
            total_cnt = 0
            case_study_dir = 'muxplm-sst2-10'
            if not os.path.exists(os.path.join('/root/mixup/mux_plms/case_study',case_study_dir)):
                os.makedirs(os.path.join('/root/mixup/mux_plms/case_study',case_study_dir))
            with open(f'/root/mixup/mux_plms/case_study/{case_study_dir}/inversion_epoch{epoch}.txt','w') as f:
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
                        f.write('-----------------------------datamux-sst2-10 case start-----------------------------\n')
                        # titile
                        f.write(f"{'origin_token':<20s} | ")
                        f.write(f"{'mux_token1':<20s}{'mux_token2':<20s}{'mux_token3':<20s}{'mux_token4':<20s}{'mux_token5':<20s}{'mux_token6':<20s}{'mux_token7':<20s}{'mux_token8':<20s}{'mux_token9':<20s}{'mux_token10':<20s} | ")
                        f.write(f"{'recover_token1':<20s}{'recover_token2':<20s}{'recover_token3':<20s}{'recover_token4':<20s}{'recover_token5':<20s}{'recover_token6':<20s}{'recover_token7':<20s}{'recover_token8':<20s}{'recover_token9':<20s}{'recover_token10':<20s}")
                        f.write('\n')
                        for word_idx in range(0, batch['input_ids'].size()[1]):
                            if valid_ids[seq_idx][word_idx]:
                                f.write(f"{tokenizer.decode(batch['input_ids'][seq_idx][word_idx]):<20s} | ")
                                for mux_idx in range(10):
                                    mux_sentence_id = batch['mux_sentence_ids'][seq_idx][mux_idx]
                                    f.write(f"{tokenizer.decode(batch['input_ids'][mux_sentence_id][word_idx]):<20s}")
                                f.write(" | ")
                                for rec_idx in range(10):
                                    f.write(f"{tokenizer.decode(top10_preds[seq_idx][word_idx][rec_idx]):<20s}")
                                # if token hited in last epoch
                                if epoch == epochs - 1: 
                                    if batch['input_ids'][seq_idx][word_idx] == top10_preds[seq_idx][word_idx][0]:
                                        hit_tokens[tokenizer.decode(batch['input_ids'][seq_idx][word_idx])] = hit_tokens.get(tokenizer.decode(batch['input_ids'][seq_idx][word_idx]), 0) + 1
                                        mux_tokens = [tokenizer.decode(batch['input_ids'][batch['mux_sentence_ids'][seq_idx][mux_idx]][word_idx]) for mux_idx in range(10)]
                                        mux_tokens_list.append(mux_tokens)
                                f.write("\n")
                        f.write('-----------------------------datamux-sst2-10 case end-----------------------------\n')
                        f.write('\n')
            model_attack_acc = hit_cnt/total_cnt
            print('attack acc:{}'.format(hit_cnt/total_cnt))
            if use_wandb:
                wandb.log({'metric/inversion_model_top{}_acc'.format(topk): hit_cnt/total_cnt})
    
    with open(f'/root/mixup/mux_plms/case_study/{case_study_dir}/hit_tokens_stat.txt','w') as f:
        hit_tokens = sorted(hit_tokens.items(), key=lambda x: x[1], reverse=True)
        for (key, value) in hit_tokens:
            f.write(f'{key:<15s}: {value}\n')
    with open(f'/root/mixup/mux_plms/case_study/{case_study_dir}/mux_tokens_stat.txt','w') as f:
        for mux_tokens in mux_tokens_list:
            for mux_token in mux_tokens:
                f.write(f"{mux_token:<15s}")
            f.write('\n')
    with open(f'/root/mixup/mux_plms/case_study/{case_study_dir}/mux_tokens_conbine_stat.txt','w') as f:
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

def knn_attack(model, dataloader, use_wandb, topk=5, target_layer=3):
    emb = model.roberta.embeddings.word_embeddings.weight.to('cpu')
    model.eval()
    hit_cnt = 0
    total_cnt = 0
    for step, batch in enumerate(dataloader):
        attention_mask = batch['attention_mask']
        valid_ids = attention_mask!=0
              
        eval_label = batch['input_ids']
        CLS_IDS = 0
        SEP_IDS = 3
        valid_ids[(eval_label==CLS_IDS) | (eval_label==SEP_IDS)] = False
        eval_label = eval_label[valid_ids] # (samples)
        preds_feature = batch['hidden_states'][valid_ids]
        ed = torch.cdist(preds_feature, emb, p=2.0) # (samples, embeddings)
        candidate_token_ids_topk = torch.topk(ed, topk, largest=False)[1] # (samples, topk)
        
        hit_cnt += (eval_label.unsqueeze(1) == candidate_token_ids_topk).int().sum().item()
        total_cnt += eval_label.shape[0]
    if use_wandb:
        wandb.log({'knn_top{}'.format(topk):hit_cnt/total_cnt})

logger = get_logger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "ag_news": ("text", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}



def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default='sst2',
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='roberta-base',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=120,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=30,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=30, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default='/root/mixup/mux_plms/ckpts', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=11, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--use_wandb",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--target_layer",
        type=int,
        default=3
    )
    parser.add_argument(
        "--wandb_name",
        default=None
    )
    parser.add_argument(
        "--mix_size",
        default=2, # {2, 4, 8, 16}
        type=int,
        help="the size of mixup",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()
        # wandb.init(config=hyperparameter_defaults, project="run_glue_hyperparam_search", entity="luyi_nlp")
    
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        if args.task_name == "ag_news":
            raw_datasets = load_dataset("ag_news")
        else:
            raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=not args.use_slow_tokenizer)
    
    config.target_layer = args.target_layer
    # config.mix_size = args.mix_size
    # config.epsilon = args.epsilon
    config.add_embedding_noise = 1
    
    # from models.modeling_roberta_mixup import RobertaForSequenceClassification
    # from models.multiplexing import RobertaSequenceClassificationMuxed
    from datamux_pretraining.models.multiplexing_pretraining_bert import MuxedBertForSequenceClassification
    model = MuxedBertForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
    
    if args.use_wandb:
        project_name = f'inversion_attack_{args.task_name}'
        wandb.init(config=config, project=project_name, entity='mixup_inference', name=args.wandb_name, sync_tensorboard=False,
                job_type="CleanRepo")
  
    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            # if label_to_id is not None:
            #     # Map labels to IDs (not necessary for GLUE tasks)
            #     result["labels"] = [label_to_id[l] for l in examples["label"]]
            # else:
            # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )
    # columns_to_return = ['input_ids', 'labels', 'attention_mask', 'cluster_ids']
    # processed_datasets.set_format(type='torch', columns=columns_to_return)

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test" if args.task_name == "ag_news" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, drop_last=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size, drop_last=True)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    if args.task_name is not None:
        if args.task_name == "ag_news":
            metric = evaluate.load("accuracy")
        else:
            metric = evaluate.load("glue", args.task_name)
    else:
        metric = evaluate.load("accuracy")
        
    # evaluation
    # eval_metric = evaluate_with_knn_attack(model, eval_dataloader, metric, accelerator, target_layer=args.target_layer, topk=10)
    # if args.use_wandb:
    #     # knn_name = 'knn_top{}'.format(knn_topk)
    #     for key,value in eval_metric.items():
    #         wandb.log({f'metric/{key}':value})
    
    model_attack_acc = train_inversion_model(config, tokenizer, model, train_dataloader, eval_dataloader, use_wandb=args.use_wandb)
    wandb.finish()
    
   
    
    

if __name__ == "__main__":
    main()