import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer, AutoConfig, get_scheduler, default_data_collator, DataCollatorForTokenClassification
from tqdm import tqdm
import random
import scipy
import math
import os
import evaluate
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.distributions.laplace import Laplace
from rouge_score import rouge_scorer
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
    def __init__(self, config, model_path_or_path='roberta-base'):
        super(InversionPLM, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_path_or_path)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, label, attention_mask=None):
        outputs = self.model(inputs_embeds=x, labels=label, attention_mask=attention_mask)
        return outputs.logits, outputs.loss

    def predict(self, x, label=None, attention_mask=None):
        outputs = self.model(inputs_embeds=x, labels=label, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(F.softmax(logits,dim=-1), dim=2)
        return logits, pred
    
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
    test_dataset = processed_datasets["test"]
    test_dataset = test_dataset.remove_columns("labels")
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=batch_size)
    return train_dataloader, eval_dataloader, test_dataloader
    

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
            target_hidden_states = outputs.hidden_states[target_layer].to('cpu')
            features.append({'hidden_states': target_hidden_states, 'input_ids': input_ids, 'attention_mask': attention_mask})
        pro_bar.update(1)
    return features

def word_filter(eval_label, filter_list):
    allow_token_ids = (eval_label == filter_list[0])
    for item in filter_list:
        allow_token_ids = allow_token_ids | (eval_label == item)
    return allow_token_ids

def test_with_knn_attack(model, dataloader, output_dir ,topk=5, target_layer=3):
    emb = model.roberta.embeddings.word_embeddings.weight
    model.eval()
    rouge_hit_cnt = 0
    top1_hit_cnt = 0
    top5_hit_cnt = 0
    total_cnt = 0
    rouge_total_cnt = 0
    predictions = []
    eval_metric = {}
    for step, batch in enumerate(dataloader):
        batch['output_hidden_states'] = True
        with torch.no_grad():
            outputs = model(**batch)
        
        prediction = outputs.logits.argmax(dim=-1)
        predictions.append(prediction)
        
        attention_mask = batch['attention_mask']
        valid_ids = attention_mask!=0
        eval_label = batch['input_ids']
        CLS_IDS = 0
        SEP_IDS = 3
        valid_ids[(eval_label==CLS_IDS) | (eval_label==SEP_IDS)] = False
        eval_label = eval_label[valid_ids] # (samples)
        preds_feature = outputs.hidden_states[target_layer][valid_ids]
        ed = torch.cdist(preds_feature, emb, p=2.0) # (samples, embeddings)
        candidate_token_ids_top1 = torch.topk(ed, 1, largest=False)[1] # (samples, topk)
        candidate_token_ids_top5 = torch.topk(ed, 5, largest=False)[1] # (samples, topk)
        
        top1_hit_cnt += (eval_label.unsqueeze(1) == candidate_token_ids_top1).int().sum().item()
        top5_hit_cnt += (eval_label.unsqueeze(1) == candidate_token_ids_top5).int().sum().item()
        total_cnt += eval_label.shape[0]
        
        # rouge
        r_hit_cnt, r_total_cnt = rouge(eval_label.unsqueeze(1), candidate_token_ids_top1, tokenizer)
        rouge_hit_cnt += r_hit_cnt
        rouge_total_cnt += r_total_cnt
    eval_metric['knn_top1'] = top1_hit_cnt/total_cnt
    eval_metric['knn_top5'] = top5_hit_cnt/total_cnt
    eval_metric['knn_rouge'] = rouge_hit_cnt/rouge_total_cnt
    
    output_predict_file = os.path.join(output_dir, f"SST-2.tsv")
    print(f"***** Predict results sst2 *****")
    with open(output_predict_file, "w") as writer:
        writer.write("index\tprediction\n")
        index=0
        for prediction in predictions:
            for item in prediction:
                writer.write(f"{index}\t{item}\n")
                index+=1
    return eval_metric
    
    

if __name__ == "__main__":
    dataset = 'sst2'
    batch_size=32 # {roberta:32, mlp:64}
    learning_rate=5e-5 # {roberta:5e-5, mlp:2e-4}
    device='cuda'
    epochs=30
    target_layer=3
    task_name='sst2'
    topk = 1
    # set hyparam
    model_path = '/root/contrastive_privacy/version_adversarial/ckpts/sst2/eps0.5_w_adversarial0.1'
    epsilon = 0.5
    wandb_name = f'best_test_noise_sst2_epsilon0.5_wadv0.1'
    wandb.init(project='sst2_privacy_test', entity='privacy_cluster', name=wandb_name)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    config = AutoConfig.from_pretrained(model_path)
    config.target_layer = target_layer
    config.epsilon = epsilon
    config.add_noise = True
    
    from models.modeling_roberta_adversarial import RobertaForSequenceClassification
    model = RobertaForSequenceClassification.from_pretrained(model_path, config=config)

    # Load dataset
    train_dataloader, eval_dataloader, test_dataloader = bulid_dataloader(task_name)
    
    # do test with knn attack
    # Get the metric function
    print('################# do test #################')
    eval_metric = test_with_knn_attack(model, test_dataloader, output_dir=model_path, topk=10, target_layer=target_layer)
    for key,value in eval_metric.items():
        wandb.log({f'test/{key}':value})
    print('done')
    
    inversion_model = InversionPLM(config)

    inversion_model = inversion_model.to(device)
    model = model.to(device)
    
    # target_dataloader = eval_dataloader

    optimizer = torch.optim.AdamW(inversion_model.parameters(), lr=learning_rate)
    
    print('load dataloader to memory')
    train_dataloader = dataloader2memory(train_dataloader, model, target_layer)
    eval_dataloader = dataloader2memory(eval_dataloader, model, target_layer)
    test_dataloader = dataloader2memory(test_dataloader, model, target_layer)
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
    filter_tokens = list(set(special_tokens))
    
    completed_steps = 0
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
            
            wandb.log({'loss/inversion_model_loss':loss.item()}, step=completed_steps)

            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
            progress_bar.update(1)
            progress_bar.set_description('inversion_model_loss:{}'.format(loss.item()))

        # do eval
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
        print('eval top1 attack acc:{}'.format(hit_cnt/total_cnt))
        wandb.log({'metric/inversion_model_top{}_acc'.format(topk): hit_cnt/total_cnt}, step=completed_steps)
    
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
            # top1
            top1_preds = torch.topk(pred_logits, k=1)[1]
            top1_preds = top1_preds[valid_ids]
            top1_hit_cnt += (test_label.unsqueeze(1) == top1_preds).int().sum().item()
            # top5
            top5_preds = torch.topk(pred_logits, k=5)[1]
            top5_preds = top5_preds[valid_ids]
            top5_hit_cnt += (test_label.unsqueeze(1) == top5_preds).int().sum().item()
            total_cnt += test_label.shape[0]
            # rouge
            r_hit_cnt, r_total_cnt = rouge(test_label.unsqueeze(1), top1_preds, tokenizer)
            rouge_hit_cnt += r_hit_cnt
            rouge_total_cnt += r_total_cnt
        print('test top1 attack acc:{}'.format(top1_hit_cnt/total_cnt))
        print('test top5 attack acc:{}'.format(top5_hit_cnt/total_cnt))
        print('test inversion rouge attack acc:{}'.format(rouge_hit_cnt/rouge_total_cnt))
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
    wandb.log({'best/best_inversion_model_top1_acc': best_top1_acc})
    wandb.log({'best/best_inversion_model_top5_acc': best_top5_acc})
    wandb.log({'best/best_inversion_model_rouge_acc': best_rouge})
    wandb.finish()