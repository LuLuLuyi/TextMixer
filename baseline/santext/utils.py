#-*- encoding:utf-8 -*-
from tqdm import tqdm
import os
import json
import unicodedata
from collections import Counter
from datasets import load_dataset

def word_normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def get_vocab_SST2(data_dir,tokenizer,tokenizer_type="subword",embedding_type="glove"):
    vocab=Counter()
    for split in ['train','dev','test']:
        data_file_path=os.path.join(data_dir,split+".tsv")
        num_lines = sum(1 for _ in open(data_file_path, encoding='utf-8'))
        with open(data_file_path, 'r', encoding='utf-8') as csvfile:
                next(csvfile)
                for line in tqdm(csvfile,total=num_lines-1):
                    line=line.strip().split("\t")
                    text = line[0]
                    if tokenizer_type=="subword":
                        tokenized_text = tokenizer.tokenize(text)
                    elif tokenizer_type=="word":
                        tokenized_text = [token.text for token in tokenizer(text)]
                    for token in tokenized_text:
                        vocab[token]+=1
    if tokenizer_type == "subword":
        if embedding_type == "bert":
            for token in tokenizer.vocab:
                vocab[token]+=1
        if embedding_type == "roberta":
            for token in tokenizer.encoder:
                vocab[token]+=1
    return vocab

def get_vocab_CliniSTS(data_dir,tokenizer,tokenizer_type="subword"):
    vocab=Counter()
    for split in ['train','dev']:
        data_file_path=os.path.join(data_dir,split+".tsv")
        num_lines = sum(1 for _ in open(data_file_path))
        with open(data_file_path, 'r') as csvfile:
            next(csvfile)
            for line in tqdm(csvfile,total=num_lines-1):
                line = line.strip().split("\t")
                text = line[7] + " " + line[8]
                if tokenizer_type=="subword":
                    tokenized_text = tokenizer.tokenize(text)
                elif tokenizer_type=="word":
                    tokenized_text = [token.text for token in tokenizer(text)]
                for token in tokenized_text:
                    vocab[token]+=1
    if tokenizer_type == "subword":
        for token in tokenizer.vocab:
            vocab[token]+=1
    return vocab

def get_vocab_QNLI(data_dir,tokenizer,tokenizer_type="subword"):
    vocab=Counter()
    for split in ['train','dev']:
        data_file_path=os.path.join(data_dir,split+".tsv")
        num_lines = sum(1 for _ in open(data_file_path))
        with open(data_file_path, 'r') as csvfile:
            next(csvfile)
            for line in tqdm(csvfile,total=num_lines-1):
                line = line.strip().split("\t")
                text = line[1] + " " + line[2]
                if tokenizer_type=="subword":
                    tokenized_text = tokenizer.tokenize(text)
                elif tokenizer_type=="word":
                    tokenized_text = [token.text for token in tokenizer(text)]
                for token in tokenized_text:
                    vocab[token]+=1
    if tokenizer_type == "subword":
        for token in tokenizer.vocab:
            vocab[token]+=1
    return vocab

def get_vocab_conll2003(data_dir,tokenizer,tokenizer_type="subword",embedding_type="glove",):
    vocab=Counter()
    dataset = load_dataset('conll2003')
    for split in ['train','validation','test']:
        data_file = dataset[split]
        for tokens in tqdm(data_file['tokens']):
            tokenized_tokens = [tokenizer(token).text for token in tokens]
            for token in tokenized_tokens:
                vocab[token]+=1
    return vocab

def get_vocab_ontonotes(data_dir,tokenizer,tokenizer_type="subword",embedding_type="glove",):
    vocab=Counter()
    dataset = load_dataset('tner/ontonotes5')
    for split in ['train','validation','test']:
        data_file = dataset[split]
        for tokens in tqdm(data_file['tokens']):
            tokenized_tokens = [tokenizer(token).text for token in tokens]
            for token in tokenized_tokens:
                vocab[token]+=1
    return vocab

def get_vocab_ag_news(data_dir,tokenizer,tokenizer_type="subword",embedding_type="glove",):
    vocab=Counter()
    dataset = load_dataset('ag_news')
    for split in ['train','test']:
        data_file = dataset[split]
        for sequence in tqdm(data_file['text']):
            tokenized_tokens = [token.text for token in tokenizer(sequence)]
            for token in tokenized_tokens:
                vocab[token]+=1
    return vocab

def get_vocab_sst2(data_dir,tokenizer,tokenizer_type="subword",embedding_type="glove",):
    vocab=Counter()
    dataset = load_dataset('sst2')
    for split in ['train','validation','test']:
        data_file = dataset[split]
        for sequence in tqdm(data_file['sentence']):
            tokenized_tokens = [token.text for token in tokenizer(sequence)]
            for token in tokenized_tokens:
                vocab[token]+=1
    return vocab