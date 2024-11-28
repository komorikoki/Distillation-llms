from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import re
from tqdm import tqdm
import torch
from torch.optim import AdamW
import matplotlib.pyplot as plt
from torch import nn
import random

import pandas as pd

# Excelファイルの読み込み
file_path = "weatherdata.xlsx"  # ファイルパスを指定
df = pd.read_excel(file_path)

data=[]
for index, row in df.iterrows():
    datad = row.to_dict()
    data.append("The weather of " + datad['Unnamed: 1']+ " is "+datad['Unnamed: 1']+".")

device='cuda'

size=365
batch=1
epochs=5

teacher_model = AutoModelForCausalLM.from_pretrained("weather_model")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

train_data=[]
for text in tqdm(data, desc="Tokenizing dataset"):
    tokenized = tokenizer(text, padding="max_length", max_length=16, truncation=True, return_tensors="pt")
    input_ids = tokenized['input_ids'].squeeze().tolist()
    attention_mask = tokenized['attention_mask'].squeeze().tolist()
    labels = input_ids[1:] + [tokenizer.pad_token_id]
    for i in range(len(attention_mask)):
        if attention_mask[i]==0:
            labels[i]=-100
    train_data.append({"input_ids": input_ids, "labels": labels, "attention_mask":attention_mask})

input_ids = [item["input_ids"] for item in train_data]
labels = [item["labels"] for item in train_data]
attention_mask = [item["attention_mask"] for item in train_data]


sbatch = int(size/batch)

batch_train=[]
for i in range(sbatch):
    batch_input=[input_ids[i]]
    batch_train.append(batch_input)
input_ids=batch_train

batch_train=[]
for i in range(sbatch):
    batch_input=[labels[i]]
    batch_train.append(batch_input)
labels=batch_train

batch_train=[]
for i in range(sbatch):
    batch_input=[attention_mask[i]]
    batch_train.append(batch_input)
attention_mask=batch_train


input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
labels_tensor = torch.tensor(labels, dtype=torch.long)
attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)



# クロスエントロピー損失関数の設定
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
criterion.to(device)

optimizer = AdamW(teacher_model.parameters(), lr=5e-5)
input_ids_tensor=input_ids_tensor.to(device)
labels_tensor=labels_tensor.to(device)
attention_mask_tensor = attention_mask_tensor.to(device)
teacher_model.to(device)
eval_loss=0
teacher_model.train()

eval_losses = []
vocab_size = teacher_model.config.vocab_size

for j in range(epochs):
    for i in tqdm(range(sbatch)):
     
        input_ids=input_ids_tensor[i]
        labels=labels_tensor[i]
        attention_mask=attention_mask_tensor[i]
        optimizer.zero_grad()
        outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits=outputs.logits
        loss = criterion(logits[0][5], labels[0][5])
        eval_loss+= loss

        loss.backward()
        optimizer.step()
        if i%100==99:
            eval_loss /= 100
            print("eval_loss", i, ":", eval_loss)
            eval_losses.append(eval_loss.item())  # 100ステップごとにeval_lossを保存
            eval_loss=0
    print("done:", j)
    eval_loss=0


teacher_model.save_pretrained("./weather_model4")