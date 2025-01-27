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
        if isinstance(dataset[i]["dialogue"], str) and isinstance(dataset[i]["summary"], str):
            reshape_dataset[i]="C: "+dataset[i]["dialogue"]+" S: "+dataset[i]["summary"]
    reshape_dataset = [item for item in reshape_dataset if isinstance(item, str)]
    reshape_dataset = [re.sub(r'\s+', ' ', item) for item in reshape_dataset]
    reshape_dataset = [re.sub(r"[^a-zA-Z0-9 .:)!?'-]", '', item) for item in reshape_dataset]
 
    return reshape_dataset[:data_size]

def reshape_sq(dataset):
    reshape_dataset = [0] * len(dataset)
    for i in range(len(dataset)):
        reshape_dataset[i]="C: "+dataset[i]["context"]+" Q: "+dataset[i]["question"]+" A: "+dataset[i]["answers"]["text"][0]
    reshape_dataset = [item for item in reshape_dataset if item != '' and len(item) >= 50 and '@' not in item]
    reshape_dataset = [re.sub(r'\s+', ' ', item) for item in reshape_dataset]
    reshape_dataset = [re.sub(r'[^a-zA-Z0-9 .:?]', '', item) for item in reshape_dataset]
 
    return reshape_dataset[:data_size]


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
        cq_len=len(tokenizer(text[:text.find(" S:")])['input_ids'])
        tokenized = tokenizer(text, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].squeeze().tolist()
        attention_mask = tokenized['attention_mask'].squeeze().tolist()
        labels = input_ids[1:] + [tokenizer.pad_token_id]
        for i in range(min(cq_len+1, 512)):
            labels[i]=128001
        data.append({"input_ids": input_ids, "labels": labels, "attention_mask":attention_mask})
    
    return data

def make_data_sq(data):
    dataset=reshape_sq(data)
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

ds = load_dataset("knkarthick/samsum")

ds_sq = load_dataset("rajpurkar/squad")

device='cuda'
model = AutoModelForCausalLM.from_pretrained("../SQuAD/model/teacher_model")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# for name, param in model.named_parameters():
#     if 'lora_A' in name or 'lora_B' in name:
#         param.requires_grad = True

data_size = 1000
data_size_v = 100
size = int(data_size/4)
size_v = int(data_size_v/4)
train_dataset=ds["train"].shuffle(seed=42).select(range(10000))
train_dataset_sq = ds_sq["train"].shuffle(seed=42).select(range(10000))

data = make_data(train_dataset)
data_sq = make_data_sq(train_dataset_sq)

input_ids_tensor = make_tensor(data, "input_ids", size)
labels_tensor = make_tensor(data, "labels", size)
attention_mask_tensor = make_tensor(data, "attention_mask", size)
input_ids_tensor_v = make_tensor(data_sq, "input_ids", size_v)
labels_tensor_v = make_tensor(data_sq, "labels", size_v)
attention_mask_tensor_v = make_tensor(data_sq, "attention_mask", size_v)


vocab_size = model.config.vocab_size
criterion = torch.nn.CrossEntropyLoss(ignore_index=128001)

criterion.to(device)
input_ids_tensor=input_ids_tensor.to(device)
labels_tensor=labels_tensor.to(device)
attention_mask_tensor=attention_mask_tensor.to(device)
input_ids_tensor_v=input_ids_tensor_v.to(device)
labels_tensor_v=labels_tensor_v.to(device)
attention_mask_tensor_v=attention_mask_tensor_v.to(device)

model.to(device)

epochs = 3
lr=1e-5

model.train()

# for name, param in model.named_parameters():
#     print(f"{name}: requires_grad={param.requires_grad}")


for j in range(epochs):
    optimizer = AdamW(model.parameters(), lr=lr)
    for i in tqdm(range(size)):
        
        input_ids=input_ids_tensor[i]
        labels=labels_tensor[i]
        attention_mask=attention_mask_tensor[i]
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits=outputs.logits
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            k=int(i/10)
            input_ids=input_ids_tensor_v[k]
            labels=labels_tensor_v[k]
            attention_mask=attention_mask_tensor_v[k]
            optimizer.zero_grad()
            outputs=model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()
        
    print("done: ", j+1, "/", epochs)
    lr/=10

model.save_pretrained("./model/train_model_distill")





