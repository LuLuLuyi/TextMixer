# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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


import dataclasses
import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import wandb

import numpy as np

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, default_data_collator
from dataset_glue import GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)


logger = logging.getLogger(__name__)


class InversionMLP(nn.Module):
    def __init__(self, config):
        super(InversionMLP,self).__init__()

        self.vocab_size = config.vocab_size
        self.input_size = config.hidden_size
        hidden_size=2048
        
        self.model = nn.Sequential(nn.Linear(self.input_size, hidden_size), 
                                nn.ReLU(), 
                                nn.Linear(hidden_size, self.vocab_size))

        self.loss = torch.nn.CrossEntropyLoss()


    def forward(self, x, labels=None, attention_mask=None, token_type_ids=None):
        if attention_mask!=None:
            x *= attention_mask[:,:,None].repeat(1,1,x.shape[-1])
        logits = self.model(x)

        loss = None
        if labels is not None:
            active_loss  = attention_mask.view(-1) == 1
            active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.loss.ignore_index).type_as(labels)
                )
            active_logits = logits.view(-1, self.vocab_size)
            loss = self.loss(active_logits, active_labels)
        return loss

    def predict(self, x, labels=None, attention_mask=None, token_type_ids=None):
        if attention_mask!=None:
            x *= attention_mask[:,:,None].repeat(1,1,x.shape[-1])
            # x *= attention_mask[:,None, :,None].repeat(1,2,1,x.shape[-1])
        logits = self.model(x)
        # logits = self.top_classifier(logits)
        pred = torch.argmax(F.softmax(logits,dim=-1), dim=2)
        return logits, pred

class InversionPLM(nn.Module):
    def __init__(self, config, model_name_or_path='roberta-base'):
        super(InversionPLM, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, label, attention_mask=None):
        outputs = self.model(inputs_embeds=x, labels=label, attention_mask=attention_mask)
        return outputs[1], outputs[0]

    def predict(self, x, label=None, attention_mask=None):
        outputs = self.model(inputs_embeds=x, labels=label, attention_mask=attention_mask)
        logits = outputs[0]
        pred = torch.argmax(F.softmax(logits,dim=-1), dim=2)
        return logits, pred


def dataloader2memory(dataloader, model, target_layer=3, device='cuda'):
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
            target_hidden_states = outputs[2][target_layer].to('cpu')
            features.append({'hidden_states': target_hidden_states, 'input_ids': input_ids, 'attention_mask': attention_mask})
        pro_bar.update(1)
    return features

def word_filter(eval_label, filter_list):
    allow_token_ids = (eval_label == filter_list[0])
    for item in filter_list:
        allow_token_ids = allow_token_ids | (eval_label == item)
    return allow_token_ids

def train_inversion_model(config, tokenizer, model, santext_train_dataloader, santext_eval_dataloader, origin_train_dataloader, origin_eval_dataloader, use_wandb=True):
    learning_rate=5e-5 # {roberta:5e-5, mlp:2e-4}
    device='cuda'
    epochs=20
    topk = 1
    inversion_model = InversionPLM(config)

    inversion_model = inversion_model.to(device)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(inversion_model.parameters(), lr=learning_rate)
    
    print('load dataloader to memory')
    santext_train_dataloader = dataloader2memory(santext_train_dataloader, model, config.target_layer, device)
    santext_eval_dataloader = dataloader2memory(santext_eval_dataloader, model, config.target_layer, device)
    print('done')
    
    total_step = len(santext_train_dataloader) * epochs
    
    progress_bar = tqdm(range(total_step))
    special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    simple_tokens = tokenizer.convert_tokens_to_ids(['.', ',', '"', '-'])
    filter_tokens = list(set(special_tokens + simple_tokens))
    
    completed_steps = 0
    model_attack_acc = 0
    print('################# start train inversion model #################')
    for epoch in range(epochs):
        for step, (santext_batch, origin_batch) in enumerate(zip(santext_train_dataloader, origin_train_dataloader)):
            santext_batch = {key:value.to(device) for key,value in santext_batch.items()}
            origin_batch = {key:value.to(device) for key,value in origin_batch.items()}
            
            target_hidden_states = santext_batch['hidden_states']
            labels = origin_batch['input_ids']
            labels[labels == tokenizer.pad_token_id] = -100
            
            attention_mask = santext_batch['attention_mask']
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

        if True:
            hit_cnt = 0
            total_cnt = 0
            for santext_batch, origin_batch in zip(santext_eval_dataloader, origin_eval_dataloader):
                santext_batch = {key:value.to(device) for key,value in santext_batch.items()}
                origin_batch = {key:value.to(device) for key,value in origin_batch.items()}

                target_hidden_states = santext_batch['hidden_states']
                eval_label = origin_batch['input_ids']
                attention_mask = santext_batch['attention_mask']

                bsz, seq_len, dim = target_hidden_states.shape
                feature = target_hidden_states
                feature = feature.to(device)
                attention_mask = attention_mask.to(device)

                pred_logits, preds = inversion_model.predict(feature, attention_mask=attention_mask)

                valid_ids = attention_mask!=0
                
                
                valid_ids[word_filter(eval_label, filter_tokens)] = False
                eval_label = eval_label[valid_ids] 
                preds = torch.topk(pred_logits, k=topk)[1]
                preds = preds[valid_ids]
                hit_cnt += (eval_label.unsqueeze(1) == preds).int().sum().item()
                total_cnt += eval_label.shape[0]
            model_attack_acc = hit_cnt/total_cnt
            print('attack acc:{}'.format(hit_cnt/total_cnt))
            if use_wandb:
                wandb.log({'metric/inversion_model_top{}_acc'.format(topk): hit_cnt/total_cnt})
    return model_attack_acc

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    wandb_name: Optional[str] = field(
        default=None,
    )
    use_wandb: Optional[int] = field(
        default=0,
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    config.target_layer = 3
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    
    if model_args.use_wandb:
        project_name = f'privacy_santext_{data_args.task_name}'
        wandb.init(config=config, project=project_name, entity='privacy_cluster', name=model_args.wandb_name)

    # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
        if training_args.do_predict
        else None
    )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )

    # # Training
    # if training_args.do_train:
    #     trainer.train(
    #         model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    #     )
    #     trainer.save_model()
    #     # For convenience, we also re-save the tokenizer to the same directory,
    #     # so that you can share your model easily on huggingface.co/models =)
    #     if trainer.is_world_master():
    #         tokenizer.save_pretrained(training_args.output_dir)

    # # Evaluation
    # eval_results = {}
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")

    #     # Loop to handle MNLI double evaluation (matched, mis-matched)
    #     eval_datasets = [eval_dataset]
    #     if data_args.task_name == "mnli":
    #         mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
    #         eval_datasets.append(
    #             GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
    #         )

    #     for eval_dataset in eval_datasets:
    #         trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
    #         eval_result = trainer.evaluate(eval_dataset=eval_dataset)

    #         output_eval_file = os.path.join(
    #             training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
    #         )
    #         if trainer.is_world_master():
    #             with open(output_eval_file, "w") as writer:
    #                 logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
    #                 for key, value in eval_result.items():
    #                     logger.info("  %s = %s", key, value)
    #                     writer.write("%s = %s\n" % (key, value))
    #         if model_args.use_wandb:
    #             for key, value in eval_result.items():
    #                 wandb.log({f'metric/{key}':value})
    #         eval_results.update(eval_result)
            
    # build santext dataloader
    santext_train_dataloader = DataLoader(train_dataset, collate_fn=default_data_collator, batch_size=training_args.per_device_train_batch_size, shuffle=False)
    santext_eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=training_args.per_device_eval_batch_size, shuffle=False)
    # build origin text dataloader
    # Get origin datasets
    data_args.data_dir = '/root/contrastive_privacy/version_text/data/SST-2'
    origin_train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    origin_eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    origin_train_dataloader = DataLoader(origin_train_dataset, collate_fn=default_data_collator, batch_size=training_args.per_device_train_batch_size, shuffle=False)
    origin_eval_dataloader = DataLoader(origin_eval_dataset, collate_fn=default_data_collator, batch_size=training_args.per_device_eval_batch_size, shuffle=False)
    model_attack_acc = train_inversion_model(
        config, tokenizer, model, 
        santext_train_dataloader, santext_eval_dataloader, 
        origin_train_dataloader, origin_eval_dataloader, 
        use_wandb=model_args.use_wandb
    )

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            )

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
    return eval_results
   
    wandb.finish()



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
