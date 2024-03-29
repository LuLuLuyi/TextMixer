#!/usr/bin/env python
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
"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""

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
from datasets import ClassLabel, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import wandb
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from torch.distributions.laplace import Laplace
from rouge_score import rouge_scorer


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.25.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

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

def train_mlc_model(config, tokenizer, model, train_dataloader, eval_dataloader,test_dataloader, use_wandb=True, output_dir=None, inversion_epochs=20, inversion_lr=5e-5):
    device ='cuda'
    learning_rate = inversion_lr # {roberta:5e-5, mlp:2e-4}
    epochs = inversion_epochs
    inversion_model = InversionPLMMLC(config)

    inversion_model = inversion_model.to(device)
    model = model.to(device)
    
    print('load dataloader to memory')
    train_dataloader = dataloader2memory_with_word_dropout(train_dataloader, model, config.nullification_rate, config.target_layer, device)
    eval_dataloader = dataloader2memory_with_word_dropout(eval_dataloader, model, config.nullification_rate, config.target_layer, device)
    test_dataloader = dataloader2memory_with_word_dropout(test_dataloader, model, config.nullification_rate, config.target_layer, device)
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
    best_test_attack_acc = 0
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
    
        print('################# start test inversion model #################')
        # do test 
        hit_cnt = 0
        total_cnt = 0
        for batch in test_dataloader:
            batch = {key:value.to(device) for key,value in batch.items()}
            
            target_hidden_states = batch['hidden_states']
            test_label = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            pred_logits, batch_preds = inversion_model.predict(target_hidden_states, attention_mask=attention_mask)
            bsz, _ = batch_preds.shape
            batch_test_label = batch['input_ids']
            for i in range(bsz):
                preds = batch_preds[i].nonzero().squeeze(-1).unsqueeze(0)
                test_label = batch_test_label[i].unsqueeze(0)
                temp_hit, temp_total = token_hit(test_label, preds, tokenizer, special_tokens)
                hit_cnt += temp_hit
                total_cnt += temp_total
        print('test mlc attack acc:{}'.format(hit_cnt/total_cnt))
        if use_wandb:
            wandb.log({'test/mlc_model_acc': hit_cnt/total_cnt})
        if hit_cnt/total_cnt > best_test_attack_acc:
            best_test_attack_acc = hit_cnt/total_cnt
    print(f'best_mlc_model_acc:{best_test_attack_acc}')
    if use_wandb:
        wandb.log({'best/best_mlc_model_acc': best_test_attack_acc})
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

def train_inversion_model(config, tokenizer, model, train_dataloader, eval_dataloader, test_dataloader, use_wandb=True, output_dir=None):
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
    test_dataloader = dataloader2memory_with_word_dropout(test_dataloader, model, config.nullification_rate, config.target_layer, device)
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

def evaluate_with_knn_attack(model, dataloader, metric, accelerator, tokenizer, label_list, nullification_rate=-1, topk=5, target_layer=3, pad_to_max_length=True):
    emb = model.roberta.embeddings.word_embeddings.weight
    model.eval()
    samples_seen = 0
    rouge_hit_cnt = 0
    hit_cnt = 0
    top1_hit_cnt = 0
    top5_hit_cnt = 0
    total_cnt = 0
    rouge_total_cnt = 0
    special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    for step, batch in enumerate(dataloader):
        nullify = (torch.rand_like(batch["attention_mask"].float()) > nullification_rate).long()
        batch["attention_mask"] = batch["attention_mask"] * nullify
        batch['output_hidden_states'] = True
        with torch.no_grad():
            outputs = model(**batch)
        # evaluate
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        if not pad_to_max_length:  # necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
        predictions_gathered, labels_gathered = accelerator.gather((predictions, labels))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(dataloader) - 1:
                predictions_gathered = predictions_gathered[: len(dataloader.dataset) - samples_seen]
                labels_gathered = labels_gathered[: len(dataloader.dataset) - samples_seen]
            else:
                samples_seen += labels_gathered.shape[0]
        preds, refs = get_labels(predictions_gathered, labels_gathered, label_list)
        metric.add_batch(
            predictions=preds,
            references=refs,
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
    eval_metric = compute_metrics(metric)
    eval_metric['knn_top{}'.format(topk)] = hit_cnt/total_cnt
    eval_metric['knn_top1'] = top1_hit_cnt/total_cnt
    eval_metric['knn_top5'] = top5_hit_cnt/total_cnt
    eval_metric['knn_rouge'] = rouge_hit_cnt/rouge_total_cnt
    return eval_metric

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='tner/ontonotes5',
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None,  help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
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
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=64,
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
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Total number of training epochs to perform.")
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
    parser.add_argument("--output_dir", type=str, default='/root/honor_dataset/save', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
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
        default=0
    )
    parser.add_argument(
        "--target_layer",
        default=3
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
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
    send_example_telemetry("run_ner_no_trainer", args)

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

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
    # 'tokens' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if "ontonotes" == args.dataset_name:
            args.dataset_name = "tner/ontonotes5"
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    if args.text_column_name is not None:
        text_column_name = args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if args.label_column_name is not None:
        label_column_name = args.label_column_name
    elif f"{args.task_name}_tags" in column_names:
        label_column_name = f"{args.task_name}_tags"
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

    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
        
    if 'ontonotes' in args.dataset_name:
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

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, num_labels=num_labels)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    if not tokenizer_name_or_path:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if config.model_type in {"bloom", "gpt2", "roberta"}:
        if 'lert' in args.model_name_or_path:
            from transformers import BertTokenzier
            tokenizer = BertTokenzier.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    
    config.target_layer = args.target_layer
    config.epsilon = args.epsilon
    config.add_noise = args.add_noise
    config.nullification_rate = args.nullification_rate
    knn_topk = 10
    
    if args.use_wandb:
        project_name = f'test_{args.dataset_name}' if 'ontonotes' not in args.dataset_name else 'test_ontonotes'
        wandb.init(config=config, project=project_name, entity='privacy_cluster', name=args.wandb_name)
    

    from models.modeling_roberta_noise import RobertaForTokenClassification
    model = RobertaForTokenClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Model has labels -> use them.
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        if list(sorted(model.config.label2id.keys())) == list(sorted(label_list)):
            # Reorganize `label_list` to match the ordering of the model.
            if labels_are_int:
                label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
                label_list = [model.config.id2label[i] for i in range(num_labels)]
            else:
                label_list = [model.config.id2label[i] for i in range(num_labels)]
                label_to_id = {l: i for i, l in enumerate(label_list)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(model.config.label2id.keys()))}, dataset labels:"
                f" {list(sorted(label_list))}.\nIgnoring the model labels as a result.",
            )

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Preprocessing the datasets.
    # First we tokenize all the ftexts.
    # padding = "max_length"
    padding = "max_length" if args.pad_to_max_length else False
    # Tokenize all texts and align the labels with them.

    
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            if 'ontonotes' in args.dataset_name:
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
                    if args.label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    with accelerator.main_process_first():
        processed_raw_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["validation"]
    test_dataset = processed_raw_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
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

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

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
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
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
        accelerator.init_trackers("ner_no_trainer", experiment_config)

    # Metrics
    metric = evaluate.load("seqeval")

    def get_labels(predictions, references):
        # Transform predictions and references tensos to numpy arrays
        if device.type == "cpu":
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

    def compute_metrics():
        results = metric.compute()
        if args.return_entity_level_metrics:
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

    # test!
    eval_metric = evaluate_with_knn_attack(model, test_dataloader, metric, accelerator, tokenizer=tokenizer, label_list=label_list, nullification_rate=config.nullification_rate, target_layer=args.target_layer, topk=knn_topk)
    if args.use_wandb:
        for key,value in eval_metric.items():
            wandb.log({f'test/{key}':value})
            
    # train mlc model
    model_attack_acc = train_mlc_model(config, tokenizer, model, train_dataloader, eval_dataloader, test_dataloader, use_wandb=args.use_wandb, output_dir=args.output_dir, inversion_epochs=20, inversion_lr=5e-5)

    # # train inversion model
    # model_attack_acc = train_inversion_model(config, tokenizer, model, train_dataloader, eval_dataloader, test_dataloader, use_wandb=args.use_wandb, output_dir=args.output_dir)
    # unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(
    #     args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    # )
    # if accelerator.is_main_process:
    #     tokenizer.save_pretrained(args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    main()