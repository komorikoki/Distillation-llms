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
        if len(tokenizer(text)['input_ids']) <= 256:
            cq_len=len(tokenizer(text[:text.find("A:")])['input_ids'])
            tokenized = tokenizer(text, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
            input_ids = tokenized['input_ids'].squeeze().tolist()
            attention_mask = tokenized['attention_mask'].squeeze().tolist()
            labels = input_ids[1:] + [tokenizer.pad_token_id]
            for i in range(min(cq_len-2, 256)):
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
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

data_size = 10000
train_dataset=ds["train"].shuffle(seed=42)

data = make_data(train_dataset)
size=int(len(data)/4)

input_ids_tensor = make_tensor(data, "input_ids", size)
labels_tensor = make_tensor(data, "labels", size)
attention_mask_tensor = make_tensor(data, "attention_mask", size)

vocab_size = model.config.vocab_size
criterion = torch.nn.CrossEntropyLoss(ignore_index=128001)

criterion.to(device)
input_ids_tensor=input_ids_tensor.to(device)
labels_tensor=labels_tensor.to(device)
attention_mask_tensor=attention_mask_tensor.to(device)
model.to(device)

epochs = 1
lr=5e-5

model.train()

for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")


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
        
    print("done: ", j+1, "/", epochs)
    lr/=10

# model.eval()
# losses=0

# for i in tqdm(range(size_v)):
    
#     input_ids_v=input_ids_tensor_v[i]
#     labels_v=labels_tensor_v[i]
#     attention_mask_v=attention_mask_tensor_v[i]
#     optimizer.zero_grad()
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids_v, attention_mask=attention_mask_v, labels=labels_v)
#         logits = outputs.logits
#     loss = criterion(logits.view(-1, vocab_size), labels_v.view(-1))
#     losses += loss

# losses=losses.item()
    
# print(f"loss train: {(losses/data_size_v):.3f}")

model.save_pretrained("./model/teacher_model")



