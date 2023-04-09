import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoModel

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
        
class InversionPLMMLC(nn.Module):
    def __init__(self, config, model_name_or_path='roberta-base'):
        super(InversionPLMMLC,self).__init__()
        self.vocab_size = config.vocab_size
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        # self.top_classifier = nn.Linear(voacb_size, voacb_size)

        self.loss = torch.nn.BCELoss()
        self.sigmod = torch.nn.Sigmoid()

    def forward(self, x, labels=None, attention_mask=None):
        logits = self.model(inputs_embeds=x, attention_mask=attention_mask).logits
        logits = self.sigmod(torch.mean(logits, dim=1))
        
        loss = None
        if labels is not None:
            valid_ids = (labels!=-100)
            active_labels = labels[valid_ids]
            active_logits = logits[valid_ids]
            active_labels = torch.nn.functional.one_hot(active_labels, num_classes=self.vocab_size)
            loss = self.loss(active_logits, labels)
        return logits, loss

    def predict(self, x, labels=None, attention_mask=None, token_type_ids=None):
        logits = self.model(inputs_embeds=x, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        logits = self.sigmod(torch.mean(logits, dim=1))
        pred = torch.round(logits)
        return logits, pred

class PositionAttackMLP(nn.Module):
    def __init__(self, config):
        super(PositionAttackMLP,self).__init__()
        
        
        self.output_size = config.max_position_embeddings
        self.input_size = config.hidden_size
        hidden_size=2048
        
        self.model = nn.Sequential(nn.Linear(self.input_size, hidden_size), 
                                nn.ReLU(), 
                                nn.Linear(hidden_size, self.output_size))

        self.loss = torch.nn.CrossEntropyLoss()


    def forward(self, x, labels=None, attention_mask=None):
        if attention_mask!=None:
            x *= attention_mask[:,:,None].repeat(1,1,x.shape[-1])
            # x *= attention_mask[:,None, :,None].repeat(1,2,1,x.shape[-1])
        logits = self.model(x)

        loss = None
        if labels is not None:
            active_loss  = attention_mask.view(-1) == 1
            active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.loss.ignore_index).type_as(labels)
                )
            active_logits = logits.view(-1, self.output_size)
            loss = self.loss(active_logits, active_labels)
        return logits, loss

    def predict(self, x, labels=None, attention_mask=None):
        if attention_mask!=None:
            x *= attention_mask[:,:,None].repeat(1,1,x.shape[-1])
            # x *= attention_mask[:,None, :,None].repeat(1,2,1,x.shape[-1])
        logits = self.model(x)
        # logits = self.top_classifier(logits)
        pred = torch.argmax(F.softmax(logits,dim=-1), dim=2)
        return logits, pred
    
class PositionAttackPLM(nn.Module):
    def __init__(self, config, model_name_or_path='roberta-base'):
        super(PositionAttackPLM, self).__init__()
        self.output_size = config.max_position_embeddings
        self.input_size = config.hidden_size
        hidden_size = 2048
        
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Sequential(nn.Linear(self.input_size, hidden_size), 
                                nn.ReLU(), 
                                nn.Linear(hidden_size, self.output_size))
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, labels, attention_mask=None):
        logits = self.model(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(logits)
        loss = None
        if labels is not None:
            active_loss  = attention_mask.view(-1) == 1
            active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.loss.ignore_index).type_as(labels)
                )
            active_logits = logits.view(-1, self.output_size)
            loss = self.loss(active_logits, active_labels)
        
        return logits, loss

    def predict(self, x, label=None, attention_mask=None):
        logits = self.model(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(logits)
        pred = torch.argmax(F.softmax(logits,dim=-1), dim=2)
        return logits, pred

   
from rouge_score import rouge_scorer

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

def token_hit(input_ids, pred_ids, tokenizer, special_tokens):
    batch_real_tokens = [tokenizer.convert_ids_to_tokens(item) for item in input_ids]
    batch_pred_tokens = [tokenizer.convert_ids_to_tokens(item) for item in pred_ids]
    hit_cnt = 0
    total_cnt = 0
    for real_tokens, pred_tokens in zip(batch_real_tokens, batch_pred_tokens):
        real_tokens = list(set(real_tokens))
        pred_tokens = list(set(pred_tokens))
        real_tokens = [token.lower() for token in real_tokens if token not in special_tokens]
        pred_tokens = [token.lower() for token in pred_tokens if token not in special_tokens]
        for token in real_tokens:
            if token in special_tokens:
                continue
            if token in pred_tokens:
                hit_cnt += 1
            total_cnt += 1
    return hit_cnt, total_cnt