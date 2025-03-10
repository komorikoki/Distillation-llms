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
student_model = AutoModelForCausalLM.from_pretrained("./model/initialized_distill_model")
teacher_model = AutoModelForCausalLM.from_pretrained("./model/teacher_model1")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

data_size = 100000
size = int(data_size/4)
train_dataset=ds["train"].shuffle(seed=42).select(range(500000))

data = make_data(train_dataset)

input_ids_tensor = make_tensor(data, "input_ids", size)
labels_tensor = make_tensor(data, "labels", size)
attention_mask_tensor = make_tensor(data, "attention_mask", size)


vocab_size = student_model.config.vocab_size
criterion = torch.nn.CrossEntropyLoss(ignore_index=128001)

criterion.to(device)
input_ids_tensor=input_ids_tensor.to(device)
labels_tensor=labels_tensor.to(device)
attention_mask_tensor=attention_mask_tensor.to(device)

epochs = 3
alpha=0.9
temperature=5
lr=1e-4
losses = 0
loss_sum = []

print("train distill")
student_model.to(device)
teacher_model.to(device)
student_model.train()
teacher_model.eval()

with open("outputtemp5.txt", "w") as f:
    for j in range(epochs):
        print(lr)
        optimizer = AdamW(student_model.parameters(), lr=lr)
        for i in tqdm(range(size)):
            optimizer.zero_grad()

            student_logits = student_model(input_ids=input_ids_tensor[i], attention_mask=attention_mask_tensor[i]).logits.view(-1, vocab_size)

            with torch.no_grad():
                teacher_logits = teacher_model(input_ids=input_ids_tensor[i], attention_mask=attention_mask_tensor[i]).logits.view(-1, vocab_size)
            mask = labels_tensor[i].view(-1) != 128001
            student_prob = F.log_softmax(student_logits[mask] / temperature, dim=-1)
            teacher_prob = F.softmax(teacher_logits[mask] / temperature, dim=-1)

            kl_loss = F.kl_div(student_prob, teacher_prob, reduction="none").sum(dim=-1).sum()
            cr_loss = criterion(student_logits.view(-1, vocab_size), labels_tensor[i].view(-1))
            kl_loss.backward()
            optimizer.step()

            losses += cr_loss.item()
            if i % 1000 == 999:
                f.write(str(round(losses/2000, 3))+"\n")
                losses = 0
            
        print("done: ", j+1, "/", epochs)
        lr/=10

student_model.save_pretrained("./model/distill_modeltemp5")


           


