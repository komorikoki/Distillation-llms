from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import optuna
import re
from tqdm import tqdm
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import AdamW
import matplotlib.pyplot as plt
from torch import nn
import gc
import sys

u=int(sys.argv[1])
print("train: ", u)

ds = load_dataset("rajpurkar/squad")
device='cuda'
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

student_model = AutoModelForCausalLM.from_pretrained("./model/testpp"+str(u))
teacher_model = AutoModelForCausalLM.from_pretrained("./model/teacher_model2")

data_size = 60000
data_size_v = 600

size=12500
size_v=100

torch.set_printoptions(profile="full")

def reshape(dataset, d_size):
    reshape_dataset = [0] * len(dataset)
    for i in range(len(dataset)):
        reshape_dataset[i]="C: "+dataset[i]["context"]+" Q: "+dataset[i]["question"]+" A: "+dataset[i]["answers"]["text"][0]
    reshape_dataset = [item for item in reshape_dataset if item != '' and len(item) >= 50 and '@' not in item]
    reshape_dataset = [re.sub(r'[^a-zA-Z0-9 .:?]', '', item) for item in reshape_dataset]
    reshape_dataset = [re.sub(r'\s+', ' ', item) for item in reshape_dataset]
    return reshape_dataset[:d_size]

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

def make_data(data, d_size):
    dataset=reshape(data, d_size)
    data = []
    for text in dataset:
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

train_dataset=ds["train"].shuffle(seed=42)

data = make_data(train_dataset, data_size)


input_ids_tensor = make_tensor(data, "input_ids", size)
labels_tensor = make_tensor(data, "labels", size)
attention_mask_tensor = make_tensor(data, "attention_mask", size)

vocab_size = teacher_model.config.vocab_size

input_ids_tensor=input_ids_tensor.to(device)
labels_tensor=labels_tensor.to(device)
attention_mask_tensor=attention_mask_tensor.to(device)



teacher_model.to(device)
teacher_model.eval()

hyps = []
loss=0
losses=[]
with open("lr.txt", "r") as f:
    lr = f.read().strip()
lr=float(lr)
# print(lr)
# lr=1e-4
temperature = 1

# print("lr", lr)

student_model.to(device)
student_model.train()

optimizer = AdamW(student_model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
for j in range(int(size/10)):
    k = int(u*(size/10) + j)

    optimizer.zero_grad()
    student_logits = student_model(input_ids=input_ids_tensor[k], attention_mask=attention_mask_tensor[k]).logits.view(-1, vocab_size)

    with torch.no_grad():
        teacher_logits = teacher_model(input_ids=input_ids_tensor[k], attention_mask=attention_mask_tensor[k]).logits.view(-1, vocab_size)
    mask = labels_tensor[k].view(-1) != 128001
    student_prob = F.log_softmax(student_logits[mask] / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits[mask] / temperature, dim=-1)

    kl_loss = F.kl_div(student_prob, teacher_prob, reduction="none").sum(dim=-1).sum()
    kl_loss.backward()
    optimizer.step()
    scheduler.step()





student_model.save_pretrained("./model/testpp"+str(u+1))
