from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import optuna
import random
import re
from tqdm import tqdm
import math
import torch
from torch.nn import functional as F
from torch.optim import AdamW
import matplotlib.pyplot as plt
from torch import nn
import gc
import sys

ds = load_dataset("rajpurkar/squad")
device='cuda'
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


tokenizer_v = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", padding_side="left")
tokenizer_v.pad_token = tokenizer_v.eos_token
tokenizer_v.pad_token_id = tokenizer_v.eos_token_id


teacher_model = AutoModelForCausalLM.from_pretrained("./model/teacher_model2")

data_size = 1000
data_size_v = 600

size=125
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
    return

def batch(input, size):
    batch_train=[]
    for i in range(size):
        batch_input=[input[4*i+0], input[4*i+1], input[4*i+2], input[4*i+3]]
        batch_train.append(batch_input)

    return batch_train

def devide(text):    
    cq = text[:(text.find("A:") + 3)]
    ans = text[(text.find("A:") + 3):]
    return [cq, ans]


# def make_data(data, d_size):
#     dataset=reshape(data, d_size)
#     data = []
#     for text in tqdm(dataset, desc="Tokenizing dataset"):
#         [cq, ans] = devide(text)
#         tokenized = tokenizer(cq, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
#         input_ids = tokenized['input_ids'].squeeze().tolist()
#         attention_mask = tokenized['attention_mask'].squeeze().tolist()
#         labels = input_ids[1:] + [tokenizer.pad_token_id]
#         ans=tokenizer(ans, truncation=True, return_tensors="pt")
#         ans = ans['input_ids'].squeeze().tolist()
#         data.append({"input_ids": input_ids, "labels": labels, "attention_mask":attention_mask, "ans":ans})
    
#     return data

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
def make_data_v(data, d_size):
    dataset=reshape(data, d_size)
    data = []
    for text in dataset:
        [cq, ans] = devide(text)
        tokenized_v = tokenizer_v(cq, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
        input_ids = tokenized_v['input_ids'].squeeze().tolist()
        attention_mask = tokenized_v['attention_mask'].squeeze().tolist()
        labels = input_ids[1:] + [tokenizer_v.pad_token_id]
        ans=tokenizer_v(ans, truncation=True, return_tensors="pt")
        ans = ans['input_ids'].squeeze().tolist()
        data.append({"input_ids": input_ids, "labels": labels, "attention_mask":attention_mask, "ans":ans})
    
    return data

def make_tensor(data, type, size):
    tmp = [item[type] for item in data]
    tmp = batch(tmp, size)
    tensor=torch.tensor(tmp, dtype=torch.long)
    return tensor

def rouge(output_ids, ans, ignore_len):
    set_ans = set(ans)
    set_ans.remove(128000)
    output = output_ids[ignore_len:].tolist()
    set_out = set(output)
    
    score =  set_out & set_ans 
    return [len(score),len(set_ans)]



# def objective(trial):
#     lr = trial.suggest_float("lr", 5e-6, 1e-4, log=True)
#     temperature = 1

#     model=AutoModelForCausalLM.from_pretrained("./model/test")
#     model.to(device)
    
#     optimizer = AdamW(model.parameters(), lr=lr)
#     losses = 0
#     acc = 0
#     words_len = 0
    

#     model.train()
#     for i in tqdm(range(size)):
#         optimizer.zero_grad()

#         student_logits = model(input_ids=input_ids_tensor[i], attention_mask=attention_mask_tensor[i]).logits.view(-1, vocab_size)

#         with torch.no_grad():
#             teacher_logits = teacher_model(input_ids=input_ids_tensor[i], attention_mask=attention_mask_tensor[i]).logits.view(-1, vocab_size)
#         mask = labels_tensor[i].view(-1) != 128001
#         student_prob = F.log_softmax(student_logits[mask] / temperature, dim=-1)
#         teacher_prob = F.softmax(teacher_logits[mask] / temperature, dim=-1)

#         kl_loss = F.kl_div(student_prob, teacher_prob, reduction="none").sum(dim=-1).sum()
#         kl_loss.backward()
#         optimizer.step()
#         del student_prob, teacher_prob

#     model.eval()
#     for i in tqdm(range(size_v)):
#         with torch.no_grad():
#             output_ids = model.generate(input_ids=input_ids_tensor_v[i], attention_mask=attention_mask_tensor_v[i], max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
#         for j in range(4):
#             rgh = rouge(output_ids[j], ans[i*4+j], ignore_len=256)
#             acc += rgh[0]
#             words_len += rgh[1]
    
#     losses = 1-(acc/words_len)     
        
#     del output_ids
#     del model, optimizer, acc, words_len
#     torch.cuda.empty_cache()
#     gc.collect()
#     return losses

# def objective_v():
#     model=AutoModelForCausalLM.from_pretrained("./model/tests")
#     model.to(device)
#     losses = 0
#     acc = 0
#     words_len = 0
#     model.eval()
#     for i in tqdm(range(size_v)):
#         with torch.no_grad():
#             output_ids = model.generate(input_ids=input_ids_tensor_v[i], attention_mask=attention_mask_tensor_v[i], max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
#         for j in range(4):
#             rgh = rouge(output_ids[j], ans[i*4+j], ignore_len=256)
#             acc += rgh[0]
#             words_len += rgh[1]

#     losses = 1-(acc/words_len)     
        
#     torch.cuda.empty_cache()
#     gc.collect()
#     print(losses)
#     return losses

train_dataset=ds["train"].shuffle(seed=40)
validation_dataset = ds["validation"].shuffle(seed=42)

data = make_data(train_dataset, data_size)
data_v = make_data_v(validation_dataset, data_size_v)


input_ids_tensor = make_tensor(data, "input_ids", size)
labels_tensor = make_tensor(data, "labels", size)
attention_mask_tensor = make_tensor(data, "attention_mask", size)
input_ids_tensor_v = make_tensor(data_v, "input_ids", size_v)
labels_tensor_v = make_tensor(data_v, "labels", size_v)
attention_mask_tensor_v = make_tensor(data_v, "attention_mask", size_v)
# ans = [data["ans"] for data in data_v]

vocab_size = teacher_model.config.vocab_size
criterion = torch.nn.CrossEntropyLoss(ignore_index=128001)

criterion.to(device)
input_ids_tensor=input_ids_tensor.to(device)
labels_tensor=labels_tensor.to(device)
attention_mask_tensor=attention_mask_tensor.to(device)
input_ids_tensor_v=input_ids_tensor_v.to(device)
labels_tensor_v=labels_tensor_v.to(device)
attention_mask_tensor_v=attention_mask_tensor_v.to(device)
ans = [datav["ans"] for datav in data_v]
teacher_model.to(device)
teacher_model.eval()

# objective_v()
loss0=0
n_trials=10
min_loss=0
best_lr=0
u=int(sys.argv[1])
temperature = 1
# # 学習率範囲（対数スケール）
# lr_min = 1e-7
# lr_max = 1e-4
lrs=[1e-4, 4.64e-5, 2.15e-5, 1e-5, 4.64e-6, 2.15e-6, 1e-6, 4.64e-7, 2.15e-7, 1e-7]
# # 対数スケールでランダムに選択
# log_lr_min = math.log10(lr_min)
# log_lr_max = math.log10(lr_max)

# model=AutoModelForCausalLM.from_pretrained("./model/test"+str(u))
# model.to(device)
# model.eval()
# for i in tqdm(range(size_v)):
#     with torch.no_grad():
#         student_logits = model(input_ids=input_ids_tensor_v[i], attention_mask=attention_mask_tensor_v[i]).logits.view(-1, vocab_size)
#         teacher_logits = teacher_model(input_ids=input_ids_tensor_v[i], attention_mask=attention_mask_tensor_v[i]).logits.view(-1, vocab_size)
#         mask = labels_tensor_v[i].view(-1) != 128001
#         student_prob = F.log_softmax(student_logits[mask] / temperature, dim=-1)
#         teacher_prob = F.softmax(teacher_logits[mask] / temperature, dim=-1)

#         kl_loss = F.kl_div(student_prob, teacher_prob, reduction="none").sum(dim=-1).sum()
#         loss0 += kl_loss.item()

# loss0 /= size_v
# print(loss0)    
# loss0=0
for m in range(n_trials):

    # random_log_lr = random.uniform(log_lr_min, log_lr_max)
    # lr = 10 ** log_lr
    # lr = round(lr, -int(math.floor(math.log10(lr))) + 3)
    lr = lrs[m]
    temperature = 1

    model=AutoModelForCausalLM.from_pretrained("./model/testkk"+str(u))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    losses = 0
    acc = 0
    words_len = 0


    model.train()
    for i in range(size):
        optimizer.zero_grad()
        student_logits = model(input_ids=input_ids_tensor[i], attention_mask=attention_mask_tensor[i]).logits.view(-1, vocab_size)
            
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids=input_ids_tensor[i], attention_mask=attention_mask_tensor[i]).logits.view(-1, vocab_size)
        mask = labels_tensor[i].view(-1) != 128001
        student_prob = F.log_softmax(student_logits[mask] / temperature, dim=-1)
        teacher_prob = F.softmax(teacher_logits[mask] / temperature, dim=-1)

        kl_loss = F.kl_div(student_prob, teacher_prob, reduction="none").sum(dim=-1).sum()
        kl_loss.backward()
        optimizer.step()
        del student_prob, teacher_prob

    model.eval()
    for i in range(size_v):
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids_tensor_v[i], attention_mask=attention_mask_tensor_v[i], max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
        
        

        for j in range(4):
            rgh = rouge(output_ids[j], ans[i*4+j], ignore_len=256)
            acc += rgh[0]
            words_len += rgh[1]

    loss0 += acc/words_len   
    print("lr:", lr, " acc:", loss0)
    if min_loss < loss0:
        min_loss = loss0
        best_lr=lr
        
    loss0=0
    # del output_ids
    # del model, optimizer, acc, words_len
    # torch.cuda.empty_cache()
    # gc.collect()


with open("lr.txt", "w") as f:
    f.write(str(best_lr))



