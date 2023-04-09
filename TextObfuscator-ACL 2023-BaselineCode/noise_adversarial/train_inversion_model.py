import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer, AutoConfig, get_scheduler, default_data_collator, DataCollatorForTokenClassification
from tqdm import tqdm
import random
import scipy
import math
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.distributions.laplace import Laplace
import wandb

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def bulid_dataloader(task_name='sst2', max_length=128):
    raw_datasets = load_dataset("glue", task_name)
    sentence1_key, sentence2_key = task_to_keys[task_name]
    padding =  "max_length" 
    max_length = 128
    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)

        if "label" in examples:
            # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )
    
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size)
    return train_dataloader, eval_dataloader
    
def bulid_dataloader_token(task_name=None, train_file='/root/honor_dataset/clear_train_1000.json', eval_file='/root/honor_dataset/test.json', max_length=128):
    data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of= None
        )
    if task_name == None:
        data_files = {}
        data_files["train"] = train_file
        data_files["validation"] = eval_file
        extension =  train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    else:
        raw_datasets = load_dataset(task_name)
    
    # raw_datasets = load_dataset("glue", task_name)
    # sentence1_key, sentence2_key = task_to_keys[task_name]
    text_column_name = 'tokens'
    padding =  False
    max_length = 128
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )
    
    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["validation"]
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)
    
    return train_dataloader, eval_dataloader

def dataloader2memory(dataloader, model, target_layer=3):
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
            target_hidden_states = outputs.hidden_states[target_layer].to('cpu')
            features.append({'hidden_states': target_hidden_states, 'input_ids': input_ids, 'attention_mask': attention_mask})
        pro_bar.update(1)
    return features

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
            # x *= attention_mask[:,None, :,None].repeat(1,2,1,x.shape[-1])
        logits = self.model(x)

        loss = None
        if labels is not None:
            bsz, seq_len, hid_dim = x.shape
            device = x.device
            active_loss  = attention_mask.view(-1) == 1
            active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.loss.ignore_index).type_as(labels)
                )
            active_logits = logits.view(-1, self.vocab_size)
            loss = self.loss(active_logits, active_labels)
        return logits, loss

    def predict(self, x, labels=None, attention_mask=None, token_type_ids=None):
        if attention_mask!=None:
            x *= attention_mask[:,:,None].repeat(1,1,x.shape[-1])
            # x *= attention_mask[:,None, :,None].repeat(1,2,1,x.shape[-1])
        logits = self.model(x)
        # logits = self.top_classifier(logits)
        pred = torch.argmax(F.softmax(logits,dim=-1), dim=2)
        return logits, pred
    
class InversionTransformer(nn.Module):
    def __init__(self, config):
        super(InversionTransformer,self).__init__()

        self.vocab_size = config.vocab_size
        self.input_size = config.hidden_size

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=8)
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.classifier = nn.Linear(self.input_size, self.vocab_size)

        self.loss = torch.nn.CrossEntropyLoss()


    def forward(self, x, labels=None, attention_mask=None, token_type_ids=None):
        # if attention_mask!=None:
        #     x *= attention_mask[:,:,None].repeat(1,1,x.shape[-1])
        # attention_mask = (-attention_mask+1).bool()
        logits = self.model(x.permute(1,0,2), src_key_padding_mask=attention_mask.bool()).permute(1,0,2)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            bsz, seq_len, hid_dim = x.shape
            device = x.device
            active_loss  = attention_mask.view(-1) == 1
            active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.loss.ignore_index).type_as(labels)
                )
            active_logits = logits.view(-1, self.vocab_size)
            loss = self.loss(active_logits, active_labels)
        return logits, loss

    def predict(self, x, labels=None, attention_mask=None, token_type_ids=None):
        logits = self.model(x.permute(1,0,2), src_key_padding_mask=attention_mask.bool()).permute(1,0,2)
        logits = self.classifier(x)
        pred = torch.argmax(F.softmax(logits,dim=-1), dim=2)
        return logits, pred  
    
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
        

def word_filter(eval_label, filter_list):
    allow_token_ids = (eval_label == filter_list[0])
    for item in filter_list:
        allow_token_ids = allow_token_ids | (eval_label == item)
    return allow_token_ids
    
    

if __name__ == "__main__":
    dataset = 'sst2'
    model_name = '/root/contrastive_privacy/version_adversarial/ckpts/sst2/eps0.5_w_adversarial0.1'
    batch_size=32
    learning_rate=5e-6
    device='cuda'
    epochs=50
    target_layer=3
    task_name='sst2'
    topk = 1
    add_noise = True
    epsilon = 0.5
    wandb_name = f'*invertionPLM_noise_adversarial_sst2_eps0.5_w_adversarial0.1_lr5e-6'
    # dataset_dir = '/root/contrastive_privacy/version1/save/data/conll2003/roberta-base_layer3/noise1_naway1_caway1_close1'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # from models.modeling_roberta_contrastive_version0_1 import RobertaForSequenceClassification, RobertaForTokenClassification
    from transformers import RobertaForSequenceClassification
    # model = RobertaForTokenClassification.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, config=config)
    

    wandb.init(project='inversion_model', entity='privacy_cluster', name=wandb_name)

    inversion_model = InversionPLM(config)

    inversion_model = inversion_model.to(device)
    model = model.to(device)
    
    # target_dataloader = eval_dataloader

    optimizer = torch.optim.AdamW(inversion_model.parameters(), lr=learning_rate)
    
    # Load dataset
    train_dataloader, eval_dataloader = bulid_dataloader(task_name)
    print('load dataloader to memory')
    train_dataloader = dataloader2memory(train_dataloader, model, target_layer)
    eval_dataloader = dataloader2memory(eval_dataloader, model, target_layer)
    print('done')
    
    total_step = len(train_dataloader) * epochs
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_step,
    )

    progress_bar = tqdm(range(total_step))

    dropout = torch.nn.Dropout(0.1)

    
    special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    simple_tokens = tokenizer.convert_tokens_to_ids(['.', ',', '"', '-'])
    filter_tokens = list(set(special_tokens + simple_tokens))
    
    completed_steps = 0
    print('################# start train inversion model #################')
    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            # input_embeddings = embedding[batch['input_ids']].clone().detach()
            
            batch = {key:value.to(device) for key,value in batch.items()}
            # batch['output_hidden_states'] = True
            # outputs = model(**batch)
            # target_hidden_states = outputs.hidden_states[target_layer]
            target_hidden_states = batch['hidden_states']
            if add_noise:
                target_noise = Laplace(loc=torch.tensor(0, device='cuda', dtype=float), scale=torch.tensor(1/epsilon, device='cuda', dtype=float)).sample(target_hidden_states.shape).type_as(target_hidden_states)
                hidden_states_min = torch.min(target_hidden_states, dim=-1, keepdim=True)[0]
                hidden_states_max = torch.max(target_hidden_states, dim=-1, keepdim=True)[0]
                target_hidden_states = (target_hidden_states - hidden_states_min) / (hidden_states_max - hidden_states_min)
                target_hidden_states =  target_hidden_states + target_noise
            labels = batch['input_ids']
            labels[labels == tokenizer.pad_token_id] = -100
            
            attention_mask = batch['attention_mask']
            
            
            bsz, seq_len, dim = target_hidden_states.shape
            feature = target_hidden_states
            
            feature = feature.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            logits, loss = inversion_model(feature, labels, attention_mask=attention_mask)
            
            wandb.log({'loss':loss.item()}, step=completed_steps)

            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
            progress_bar.update(1)
            progress_bar.set_description('loss:{}'.format(loss.item()))

        if True:
            hit_cnt = 0
            total_cnt = 0
            for batch in eval_dataloader:
                batch = {key:value.to(device) for key,value in batch.items()}
                # batch['output_hidden_states'] = True
                
                # outputs = model(**batch)
                # target_hidden_states = outputs.hidden_states[target_layer]
                target_hidden_states = batch['hidden_states']
                if add_noise:
                    target_noise = Laplace(loc=torch.tensor(0, device='cuda', dtype=float), scale=torch.tensor(1/epsilon, device='cuda', dtype=float)).sample(target_hidden_states.shape).type_as(target_hidden_states)
                    hidden_states_min = torch.min(target_hidden_states, dim=-1, keepdim=True)[0]
                    hidden_states_max = torch.max(target_hidden_states, dim=-1, keepdim=True)[0]
                    target_hidden_states = (target_hidden_states - hidden_states_min) / (hidden_states_max - hidden_states_min)
                    target_hidden_states =  target_hidden_states + target_noise
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
            print('attack acc:{}'.format(hit_cnt/total_cnt))
            wandb.log({'top{}_acc'.format(topk): hit_cnt/total_cnt}, step=completed_steps)
    wandb.finish()