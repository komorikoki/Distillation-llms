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
student_model = AutoModelForCausalLM.from_pretrained("../model/LoRA_distill_model")
teacher_model = AutoModelForCausalLM.from_pretrained("./model/teacher_model")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

for name, param in student_model.named_parameters():
    if not 'base' in name:
        param.requires_grad = True
        # if 'self' in name:
        #     param.requires_grad = True

data_size = 50000

train_dataset=ds["train"].shuffle(seed=42)

data = make_data(train_dataset)
size=int(len(data)/4)
input_ids_tensor = make_tensor(data, "input_ids", size)
labels_tensor = make_tensor(data, "labels", size)
attention_mask_tensor = make_tensor(data, "attention_mask", size)


vocab_size = student_model.config.vocab_size
criterion = torch.nn.CrossEntropyLoss(ignore_index=128001)

criterion.to(device)
input_ids_tensor=input_ids_tensor.to(device)
labels_tensor=labels_tensor.to(device)
attention_mask_tensor=attention_mask_tensor.to(device)

student_model.to(device)
teacher_model.to(device)

epochs = 3
lr=5e-5
temperature = 1

student_model.train()
teacher_model.eval()

for name, param in student_model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

for j in range(epochs):
    optimizer = AdamW(student_model.parameters(), lr=lr)
    for i in tqdm(range(size)):
        input_ids=input_ids_tensor[i]
        labels=labels_tensor[i]
        attention_mask=attention_mask_tensor[i]
        optimizer.zero_grad()
        student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits
        student_prob=F.log_softmax(student_logits, dim=-1)
        student_prob_view = student_prob.view(-1, vocab_size)
        

        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits
            teacher_prob = F.softmax(teacher_logits, dim=-1)
            teacher_prob_view = teacher_prob.view(-1, vocab_size)
        

        sec_student_prob=[]
        sec_teacher_prob=[]
        for i in range(labels.view(-1).size(0)):
            if labels.view(-1)[i] == 128001:
                sec_student_prob.append(torch.zeros_like(student_prob_view[i]))  
                sec_teacher_prob.append(torch.zeros_like(teacher_prob_view[i]))
            else:
                sec_student_prob.append(student_prob_view[i])  
                sec_teacher_prob.append(teacher_prob_view[i])
        sec_student = torch.stack(sec_student_prob, dim=0)
        sec_teacher = torch.stack(sec_teacher_prob, dim=0)

        kldiv_loss=F.kl_div(sec_student/temperature, sec_teacher/temperature, reduction="none")
        kl_div_answer = kldiv_loss.sum(dim=-1)
        kl_loss=kl_div_answer.sum()

        loss= kl_loss
        print(loss)
        loss.backward()
        optimizer.step()

        # if size == 100:
        #     lr/=5
        #     optimizer = AdamW(student_model.parameters(), lr=lr)

        
    print("done: ", j+1, "/", epochs)
    lr/= 5


student_model.save_pretrained("./model/distill_distilledmodel4")



