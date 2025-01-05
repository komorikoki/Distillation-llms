from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import re
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.optim import AdamW
import matplotlib.pyplot as plt
from torch import nn



def reshape(dataset):
    reshape_dataset = [0] * len(dataset)
    for i in range(len(dataset)):
        reshape_dataset[i]="C: "+dataset[i]["context"]+" Q: "+dataset[i]["question"]+" A: "+dataset[i]["answers"]["text"][0]
    reshape_dataset = [item for item in reshape_dataset if item != '' and len(item) >= 50 and '@' not in item]
    reshape_dataset = [re.sub(r'[^a-zA-Z0-9 .:?]', '', item) for item in reshape_dataset]
    reshape_dataset = [re.sub(r'\s+', ' ', item) for item in reshape_dataset]
    return reshape_dataset[:data_size_v]

def max_length(dataset):
    max_eval=0
    for i in dataset:
        max_eval = len(i) if len(i) > max_eval else max_eval
    print(max_eval)
    return

def batch(input, size):
    batch_train=[]
    for i in range(size):
        batch_input=[input[4*i+0], input[4*i+1], input[4*i+2], input[4*i+3]]
        batch_train.append(batch_input)

    return batch_train

def make_data(data):
    dataset=reshape(data)
    data = []
    for text in tqdm(dataset, desc="Tokenizing dataset"):
        cq_len=len(tokenizer(text[:text.find("A:")])['input_ids'])
        tokenized = tokenizer(text, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].squeeze().tolist()
        attention_mask = tokenized['attention_mask'].squeeze().tolist()
        labels = input_ids[1:] + [tokenizer.pad_token_id]
        for i in range(min(cq_len-2, 512)):
            labels[i]=128001
        data.append({"input_ids": input_ids, "labels": labels, "attention_mask":attention_mask})
    
    return data

def make_tensor(data, type, size):
    tmp = [item[type] for item in data]
    tmp = batch(tmp, size)
    tensor=torch.tensor(tmp, dtype=torch.long)
    return tensor

ds = load_dataset("rajpurkar/squad")
device='cuda'
model = AutoModelForCausalLM.from_pretrained("../model/QLoRA_distill_model")
# model = AutoModelForCausalLM.from_pretrained("./model/normal_model_QLoRA2")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

data_size_v = 1000
size_v = int(data_size_v/4)
validation_dataset=ds["validation"].shuffle(seed=42)

data_v = make_data(validation_dataset)

input_ids_tensor_v = make_tensor(data_v, "input_ids", size_v)
labels_tensor_v = make_tensor(data_v, "labels", size_v)
attention_mask_tensor_v = make_tensor(data_v, "attention_mask", size_v)


vocab_size = model.config.vocab_size
criterion = torch.nn.CrossEntropyLoss(ignore_index=128001)

criterion.to(device)

input_ids_tensor_v=input_ids_tensor_v.to(device)
labels_tensor_v=labels_tensor_v.to(device)
attention_mask_tensor_v=attention_mask_tensor_v.to(device)

model.to(device)

model.eval()
losses=0

for i in tqdm(range(size_v)):
    
    input_ids_v=input_ids_tensor_v[i]
    labels_v=labels_tensor_v[i]
    attention_mask_v=attention_mask_tensor_v[i]
    with torch.no_grad():
        outputs = model(input_ids=input_ids_v, attention_mask=attention_mask_v, labels=labels_v)
        logits = outputs.logits
    loss = criterion(logits.view(-1, vocab_size), labels_v.view(-1))
    losses += loss

losses=losses.item()
    
print(f"loss train: {(losses/data_size_v):.3f}")




