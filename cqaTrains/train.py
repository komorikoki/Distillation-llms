from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ds = load_dataset("rajpurkar/squad")


import re
from tqdm import tqdm
import torch
from torch.optim import AdamW
import matplotlib.pyplot as plt
from torch import nn
import random

import pandas as pd


device='cuda'

size=10000
batch=4
epochs=5

model = AutoModelForCausalLM.from_pretrained("../distillLLAMA2")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
train_dataset=ds["train"].shuffle(seed=42).select(range(size))

train_data=[]
for i in range(size):
    tokenized = tokenizer(train_dataset[i]['context']+train_dataset[i]['question'] + train_dataset[i]['answers']['text'][0], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    cqlen = len(tokenizer(train_dataset[i]['context']+train_dataset[i]['question'])['input_ids'])
    length=torch.sum(tokenized['attention_mask'][0])
    input_ids = tokenized['input_ids'].squeeze().tolist()
    attention_mask = tokenized['attention_mask'].squeeze().tolist()
    labels = input_ids[1:] + [tokenizer.pad_token_id]
    for i in range(len(attention_mask)):
        if attention_mask[i]==0:
            labels[i]=-100

    # labels=[-100]*512
    # for i in range(length-cqlen):
    #     labels[cqlen+i-1]=input_ids[cqlen+i]
    train_data.append({"input_ids": input_ids, "labels": labels, "attention_mask":attention_mask})

sbatch = int(size/batch)


input_ids = [item["input_ids"] for item in train_data]
labels = [item["labels"] for item in train_data]
attention_mask = [item["attention_mask"] for item in train_data]


batch_train=[]
for i in range(sbatch):
    batch_input=[input_ids[i], input_ids[i+1], input_ids[i+2], input_ids[i+3]]
    batch_train.append(batch_input)
input_ids=batch_train

batch_train=[]
for i in range(sbatch):
    batch_input=[labels[i], labels[i+1], labels[i+2], labels[i+3]]
    batch_train.append(batch_input)
labels=batch_train

batch_train=[]
for i in range(sbatch):
    batch_input=[attention_mask[i], attention_mask[i+1], attention_mask[i+2], attention_mask[i+3]]
    batch_train.append(batch_input)
attention_mask=batch_train

input_ids = [item["input_ids"] for item in train_data]
labels = [item["labels"] for item in train_data]
attention_mask = [item["attention_mask"] for item in train_data]


input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
labels_tensor = torch.tensor(labels, dtype=torch.long)
attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)


# クロスエントロピー損失関数の設定
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
criterion.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
input_ids_tensor=input_ids_tensor.to(device)
labels_tensor=labels_tensor.to(device)
attention_mask_tensor = attention_mask_tensor.to(device)
model.to(device)
eval_loss=0
model.train()

eval_losses = []
vocab_size = model.config.vocab_size

for j in range(epochs):
    for i in tqdm(range(sbatch)):
     
        input_ids=input_ids_tensor[i].unsqueeze(0)
        labels=labels_tensor[i].unsqueeze(0)
        attention_mask=attention_mask_tensor[i].unsqueeze(0)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits=outputs.logits
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        loss
        eval_loss+= loss
        loss.backward()
        optimizer.step()
        if i%100==99:
            eval_loss /= 100
            print("eval_loss", i,":", eval_loss.item())
            eval_loss=0
    print("done:", j,"/",epochs)
    eval_loss=0


model.save_pretrained("./cqatrains1")