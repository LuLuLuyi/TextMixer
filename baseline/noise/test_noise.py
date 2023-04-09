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
import wandb
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
from rouge_score import rouge_scorer


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

class InversionPLMMLC(nn.Module):
    def __init__(self, config, model_name_or_path='roberta-base'):
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
   
# class InversionPLMMLC(nn.Module):
#     def __init__(self, config, model_name_or_path='roberta-base'):
#         super(InversionPLMMLC,self).__init__()
#         self.vocab_size = config.vocab_size
#         self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

#         self.loss = torch.nn.BCELoss()
#         self.sigmod = torch.nn.Sigmoid()

#     def forward(self, x, labels=None, attention_mask=None, token_type_ids=None):
#         bsz, seq_len, hid_dim = x.shape
#         device = x.device
    
#         logits = self.model(inputs_embeds=x, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        
#         logits = logits * attention_mask.unsqueeze(-1)
#         logits = logits.sum(dim=1)/attention_mask.sum(dim=1).unsqueeze(-1)
        
#         logits = self.sigmod(logits) # (bsz, dim)

#         loss = None
#         if labels is not None:
#             labels = torch.zeros(bsz, self.vocab_size).to(device).scatter_(1, labels, 1.)
#             labels[:,0:3] = 0
#             loss = self.loss(logits, labels)
#         return logits, loss

#     def predict(self, x, labels=None, attention_mask=None, token_type_ids=None):
#         logits = self.model(inputs_embeds=x, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        
#         logits = logits * attention_mask.unsqueeze(-1)
#         logits = logits.sum(dim=1)/attention_mask.sum(dim=1).unsqueeze(-1)
#         logits = self.sigmod(logits)
        
#         threshold = 0.5
#         pred = (logits > threshold).long()
#         return logits, pred
    
def token_hit(input_ids, pred_ids, tokenizer, special_tokens):
    batch_real_tokens = [tokenizer.convert_ids_to_tokens(item) for item in input_ids]
    batch_pred_tokens = [tokenizer.convert_ids_to_tokens(item) for item in pred_ids]
    hit_cnt = 0
    total_cnt = 0
    special_tokens = ['<s>', '</s>', '<pad>']
    for real_tokens, pred_tokens in zip(batch_real_tokens, batch_pred_tokens):
        real_tokens = {item.replace('Ġ', '').lower() for item in set(real_tokens) if item not in special_tokens}
        pred_tokens = {item.replace('Ġ', '').lower() for item in set(pred_tokens) if item not in special_tokens}
        hit_cnt += len(real_tokens & pred_tokens)
        total_cnt += len(real_tokens)
    return hit_cnt, total_cnt

def train_mlc_model(config, tokenizer, model, train_dataloader, eval_dataloader, use_wandb=True, output_dir=None, inversion_epochs=20, inversion_lr=5e-5):
    device ='cuda'
    learning_rate = inversion_lr # {roberta:5e-5, mlp:2e-4}
    epochs = inversion_epochs
    inversion_model = InversionPLMMLC(config)

    inversion_model = inversion_model.to(device)
    model = model.to(device)
    
    print('load dataloader to memory')
    train_dataloader = dataloader2memory_with_word_dropout(train_dataloader, model, config.nullification_rate, config.target_layer, device)
    eval_dataloader = dataloader2memory_with_word_dropout(eval_dataloader, model, config.nullification_rate, config.target_layer, device)
    print('done')
    
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
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_step,
    )
    
    progress_bar = tqdm(range(total_step))
    special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    filter_tokens = list(set(special_tokens))
    
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
                    temp_hit, temp_total = token_hit(eval_label, preds, tokenizer, special_tokens)
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
    return model_attack_acc

def dataloader2memory_with_word_dropout(dataloader, model, nullification_rate, target_layer=3, device='cuda'):
    features = []
    pro_bar = tqdm(range(len(dataloader)))
    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            batch = {key:value.to(device) for key,value in batch.items()}
            batch['output_hidden_states'] = True
            outputs = model(**batch)
            input_ids = batch['input_ids'].to('cpu')
            attention_mask = batch['attention_mask'].to('cpu')
            # add word dropout
            nullify = (torch.rand_like(attention_mask.float()) > nullification_rate).long()
            attention_mask = attention_mask * nullify
            target_hidden_states = outputs.hidden_states[target_layer].to('cpu')
            features.append({'hidden_states': target_hidden_states, 'input_ids': input_ids, 'attention_mask': attention_mask})
        pro_bar.update(1)
    return features

def word_filter(eval_label, filter_list):
    allow_token_ids = (eval_label == filter_list[0])
    for item in filter_list:
        allow_token_ids = allow_token_ids | (eval_label == item)
    return allow_token_ids


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


def train_inversion_model(config, tokenizer, model, train_dataloader, eval_dataloader, use_wandb=True, output_dir=None):
    learning_rate=5e-5 # {roberta:5e-5, mlp:2e-4}
    device='cuda'
    epochs=30
    topk = 1
    inversion_model = InversionPLM(config)

    inversion_model = inversion_model.to(device)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(inversion_model.parameters(), lr=learning_rate)
    
    print('load dataloader to memory')
    train_dataloader = dataloader2memory_with_word_dropout(train_dataloader, model, config.nullification_rate, config.target_layer, device)
    eval_dataloader = dataloader2memory_with_word_dropout(eval_dataloader, model, config.nullification_rate, config.target_layer, device)
    test_dataloader = eval_dataloader
    print('done')
    
    total_step = len(train_dataloader) * epochs
    
    progress_bar = tqdm(range(total_step))
    special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    filter_tokens = list(set(special_tokens))
    
    completed_steps = 0
    model_attack_acc = 0
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
            
            feature = target_hidden_states
            
            feature = feature.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            logits, loss = inversion_model(feature, labels, attention_mask=attention_mask)
            if use_wandb:
                wandb.log({'loss/inversion_model_loss':loss.item()})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            completed_steps += 1
            progress_bar.update(1)
            progress_bar.set_description('inversion_model_loss:{}'.format(loss.item()))

        hit_cnt = 0
        total_cnt = 0
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
            preds = torch.topk(pred_logits, k=topk)[1]
            preds = preds[valid_ids]
            hit_cnt += (eval_label.unsqueeze(1) == preds).int().sum().item()
            total_cnt += eval_label.shape[0]
        model_attack_acc = hit_cnt/total_cnt
        print('attack acc:{}'.format(hit_cnt/total_cnt))
        if use_wandb:
            wandb.log({'metric/inversion_model_top{}_acc'.format(topk): hit_cnt/total_cnt})

        print('################# start test inversion model #################')
        # do test 
        rouge_hit_cnt = 0
        top1_hit_cnt = 0
        top5_hit_cnt = 0
        total_cnt = 0
        rouge_total_cnt = 0
        for batch in test_dataloader:
            batch = {key:value.to(device) for key,value in batch.items()}
            target_hidden_states = batch['hidden_states']
            
            test_label = batch['input_ids']
            attention_mask = batch['attention_mask']

            feature = target_hidden_states
            feature = feature.to(device)
            attention_mask = attention_mask.to(device)
            pred_logits, preds = inversion_model.predict(feature, attention_mask=attention_mask)

            valid_ids = attention_mask!=0 
            valid_ids[word_filter(test_label, filter_tokens)] = False
            test_label = batch['input_ids']
            test_label = test_label[valid_ids] 
            # inversion top1
            top1_preds = torch.topk(pred_logits, k=1)[1]
            top1_preds = top1_preds[valid_ids]
            top1_hit_cnt += (test_label.unsqueeze(1) == top1_preds).int().sum().item()
            # inversion top5
            top5_preds = torch.topk(pred_logits, k=5)[1]
            top5_preds = top5_preds[valid_ids]
            top5_hit_cnt += (test_label.unsqueeze(1) == top5_preds).int().sum().item()
            total_cnt += test_label.shape[0]
            # rouge
            r_hit_cnt, r_total_cnt = rouge(test_label.unsqueeze(1), top1_preds, tokenizer)
            rouge_hit_cnt += r_hit_cnt
            rouge_total_cnt += r_total_cnt
            
        print('test inversion top1 attack acc:{}'.format(top1_hit_cnt/total_cnt))
        print('test inversion top5 attack acc:{}'.format(top5_hit_cnt/total_cnt))
        print('test inversion rouge attack acc:{}'.format(rouge_hit_cnt/rouge_total_cnt))
        if use_wandb:
            wandb.log({'test/inversion_model_top1_acc': top1_hit_cnt/total_cnt})
            wandb.log({'test/inversion_model_top5_acc': top5_hit_cnt/total_cnt})
            wandb.log({'test/inversion_model_rouge_acc': rouge_hit_cnt/rouge_total_cnt})
        if top1_hit_cnt/total_cnt > best_top1_acc:
            best_top1_acc = top1_hit_cnt/total_cnt
        if top5_hit_cnt/total_cnt > best_top5_acc:
            best_top5_acc = top5_hit_cnt/total_cnt
        if rouge_hit_cnt/rouge_total_cnt > best_rouge:
            best_rouge = rouge_hit_cnt/rouge_total_cnt
    print(f'best_inversion_model_top1_acc:{best_top1_acc}')
    print(f'best_inversion_model_top5_acc:{best_top5_acc}')
    print(f'best_inversion_model_rouge:{best_rouge}')
    if use_wandb:
        wandb.log({'best/best_inversion_model_top1_acc': best_top1_acc})
        wandb.log({'best/best_inversion_model_top5_acc': best_top5_acc})
        wandb.log({'best/best_inversion_model_rouge_acc': best_rouge})
    # save inversion model
    torch.save(inversion_model, os.path.join(output_dir,'inversion_model.pt'))
    return model_attack_acc




def evaluate_with_knn_attack(model, tokenizer, dataloader, metric, accelerator, nullification_rate=-1, topk=5, target_layer=3):
    emb = model.roberta.embeddings.word_embeddings.weight
    model.eval()
    samples_seen = 0
    rouge_hit_cnt = 0
    hit_cnt = 0
    top1_hit_cnt = 0
    top5_hit_cnt = 0
    total_cnt = 0
    rouge_total_cnt = 0
    for step, batch in enumerate(dataloader):
        nullify = (torch.rand_like(batch["attention_mask"].float()) > nullification_rate).long()
        batch["attention_mask"] = batch["attention_mask"] * nullify
        batch['output_hidden_states'] = True
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
        
        attention_mask = batch['attention_mask']
        valid_ids = attention_mask!=0
        eval_label = batch['input_ids']
        CLS_IDS = 0
        SEP_IDS = 3
        valid_ids[(eval_label==CLS_IDS) | (eval_label==SEP_IDS)] = False
        eval_label = eval_label[valid_ids] # (samples)
        preds_feature = outputs.hidden_states[target_layer][valid_ids]
        ed = torch.cdist(preds_feature, emb, p=2.0) # (samples, embeddings)
        candidate_token_ids_topk = torch.topk(ed, topk, largest=False)[1] # (samples, topk)
        candidate_token_ids_top1 = torch.topk(ed, 1, largest=False)[1] # (samples, topk)
        candidate_token_ids_top5 = torch.topk(ed, 5, largest=False)[1] # (samples, topk)
        hit_cnt += (eval_label.unsqueeze(1) == candidate_token_ids_topk).int().sum().item()
        top1_hit_cnt += (eval_label.unsqueeze(1) == candidate_token_ids_top1).int().sum().item()
        top5_hit_cnt += (eval_label.unsqueeze(1) == candidate_token_ids_top5).int().sum().item()
        total_cnt += eval_label.shape[0]
        # rouge
        r_hit_cnt, r_total_cnt = rouge(eval_label.unsqueeze(1), candidate_token_ids_top1, tokenizer)
        rouge_hit_cnt += r_hit_cnt
        rouge_total_cnt += r_total_cnt
        
    eval_metric = metric.compute()
    eval_metric['knn_top{}'.format(topk)] = hit_cnt/total_cnt
    eval_metric['knn_top1'] = top1_hit_cnt/total_cnt
    eval_metric['knn_top5'] = top5_hit_cnt/total_cnt
    eval_metric['knn_rouge'] = rouge_hit_cnt/rouge_total_cnt
    return eval_metric

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
        default=128,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
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
    parser.add_argument("--output_dir", type=str, default='/root/contrastive_privacy/version_noise/ckpts', help="Where to store the final model.")
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
        "--epsilon",
        default=1, # {0.05, 0.1, 0.5, 1, 5}
        type=float,
        help="DP epsilon",
    )
    parser.add_argument(
        "--add_noise",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--nullification_rate",
        default=-1,
        type=float,
        help="nullification rate",
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    
    config.target_layer = args.target_layer
    config.epsilon = args.epsilon
    config.add_noise = args.add_noise
    config.nullification_rate = args.nullification_rate
    
    if args.use_wandb:
        project_name = f'test_{args.task_name}'
        wandb.init(config=config, project=project_name, entity='privacy_cluster', name=args.wandb_name)

    from models.modeling_roberta_noise import RobertaForSequenceClassification
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
        
  
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
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

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
    
    
    # test!
    eval_metric = evaluate_with_knn_attack(model, tokenizer, eval_dataloader, metric, accelerator, nullification_rate=config.nullification_rate, topk=10, target_layer=args.target_layer)
    if args.use_wandb:
        for key,value in eval_metric.items():
            wandb.log({f'test/{key}':value})
    # # train inversion model
    # model_attack_acc = train_inversion_model(config, tokenizer, model, train_dataloader, eval_dataloader, use_wandb=args.use_wandb, output_dir=args.output_dir)
    
    # train mlc model
    model_attack_acc = train_mlc_model(config, tokenizer, model, train_dataloader, eval_dataloader, use_wandb=args.use_wandb, output_dir=args.output_dir, inversion_epochs=20, inversion_lr=5e-5)

    
    wandb.finish()
    
   
    
    

if __name__ == "__main__":
    main()