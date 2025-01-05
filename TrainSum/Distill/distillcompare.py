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
    dataset=dataset["text"]
    dataset = [item for item in dataset if item != '' and len(item) >= 50 and '@' not in item]
    dataset = [re.sub(r'[^a-zA-Z0-9 .?]', '', item) for item in dataset]
    dataset = [re.sub(r'\s+', ' ', item) for item in dataset]
    return dataset[:data_size]

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
        tokenized = tokenizer(text, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].squeeze().tolist()
        attention_mask = tokenized['attention_mask'].squeeze().tolist()
        labels = input_ids[1:] + [tokenizer.pad_token_id]
        data.append({"input_ids": input_ids, "labels": labels, "attention_mask":attention_mask})
    
    return data

def make_tensor(data, type, size):
    tmp = [item[type] for item in data]
    tmp = batch(tmp, size)
    tensor=torch.tensor(tmp, dtype=torch.long)
    return tensor

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
device='cuda'
model = AutoModelForCausalLM.from_pretrained("./model/initialized_distill_model2")
student_model = AutoModelForCausalLM.from_pretrained("./model/initialized_distill_model2")
teacher_model = AutoModelForCausalLM.from_pretrained("./model/teacher_model1")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

data_size = 100000
data_size_v = 766
size = int(data_size/4)
size_v = int(data_size_v/4)
train_dataset=ds["train"].shuffle(seed=42).select(range(500000))
validation_dataset=ds["validation"].shuffle(seed=42)

data = make_data(train_dataset)
data_v = make_data(validation_dataset)

input_ids_tensor = make_tensor(data, "input_ids", size)
labels_tensor = make_tensor(data, "labels", size)
attention_mask_tensor = make_tensor(data, "attention_mask", size)
input_ids_tensor_v = make_tensor(data_v, "input_ids", size_v)
labels_tensor_v = make_tensor(data_v, "labels", size_v)
attention_mask_tensor_v = make_tensor(data_v, "attention_mask", size_v)


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
lr=1e-4

print("train normal")

model.train()

for j in range(epochs):
    print(lr)
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
print("eval normal")

model.eval()
losses=0
for i in tqdm(range(size_v)):
    
    input_ids_v=input_ids_tensor_v[i]
    labels_v=labels_tensor_v[i]
    attention_mask_v=attention_mask_tensor_v[i]
    optimizer.zero_grad()
    with torch.no_grad():
        outputs = model(input_ids=input_ids_v, attention_mask=attention_mask_v, labels=labels_v)
        logits = outputs.logits
    loss = criterion(logits.view(-1, vocab_size), labels_v.view(-1))
    losses += loss

losses=losses.item()
    
print(f"loss train: {(losses/data_size_v):.3f}")

model.save_pretrained("./model/normal_model")

alpha=0.9
temperature=20
lr=1e-4

print("train distill")
student_model.to(device)
teacher_model.to(device)
student_model.train()
teacher_model.eval()

for j in range(epochs):
    print(lr)
    optimizer = AdamW(student_model.parameters(), lr=lr)
    for i in tqdm(range(size)):
        
        input_ids=input_ids_tensor[i]
        labels=labels_tensor[i]
        attention_mask=attention_mask_tensor[i]
        optimizer.zero_grad()
        outputs_st = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits_st=outputs_st.logits
        student_prob=F.log_softmax(logits_st, dim=-1)
        with torch.no_grad():
            teacher_outputs_logits=teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
            teacher_prob=F.softmax(teacher_outputs_logits, dim=-1)
        kldiv_loss=F.kl_div(student_prob/temperature, teacher_prob/temperature, reduction="none")   
        kl_div_per_token = kldiv_loss.sum(dim=-1)
        kl_loss=(kl_div_per_token * attention_mask).sum()/attention_mask.sum()

        cr_loss = criterion(logits_st.view(-1, vocab_size), labels.view(-1))
        # loss = (1-alpha)*cr_loss + alpha * kl_loss*10
        loss = kl_loss * 10

        loss.backward()
        optimizer.step()
        
    print("done: ", j+1, "/", epochs)
    lr/=10

print("eval distill")

student_model.eval()
losses_d=0

for i in tqdm(range(size_v)):
    
    input_ids_v=input_ids_tensor_v[i]
    labels_v=labels_tensor_v[i]
    attention_mask_v=attention_mask_tensor_v[i]
    optimizer.zero_grad()
    with torch.no_grad():
        outputs = student_model(input_ids=input_ids_v, attention_mask=attention_mask_v, labels=labels_v)
        logits = outputs.logits
    loss = criterion(logits.view(-1, vocab_size), labels_v.view(-1))
    losses_d += loss
    
losses_d=losses_d.item()
print(f"loss distill: {(losses_d/data_size_v):.3f}")




print(f"normal loss:{(losses/data_size_v):.3f} distill loss:{(losses_d/data_size_v):.3f}")


student_model.save_pretrained("./model/distill_model")



