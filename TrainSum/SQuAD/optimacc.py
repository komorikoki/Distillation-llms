from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import re
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.optim import AdamW
import matplotlib.pyplot as plt
import sys
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

def devide(text):    
    cq = text[:(text.find("A:") + 3)]
    ans = text[(text.find("A:") + 3):]
    return [cq, ans]

def make_data(data):
    dataset=reshape(data)
    data = []
    for text in dataset:
        [cq, ans] = devide(text)
        tokenized = tokenizer(cq, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].squeeze().tolist()
        attention_mask = tokenized['attention_mask'].squeeze().tolist()
        labels = input_ids[1:] + [tokenizer.pad_token_id]
        ans=tokenizer(ans, truncation=True, return_tensors="pt")
        ans = ans['input_ids'].squeeze().tolist()
        data.append({"input_ids": input_ids, "labels": labels, "attention_mask":attention_mask, "ans":ans})
    
    return data

def make_tensor(data, type, size):
    tmp = [item[type] for item in data]
    tmp = batch(tmp, size)
    tensor=torch.tensor(tmp, dtype=torch.long)
    return tensor

def accuracy(output_ids, ans, ignore_len):
    data_num=0
    acc_num=0
    for i in range(len(ans)-1):
        data_num += 1
        if output_ids[ignore_len+i]==ans[i+1]:
            
            acc_num += 1   
    return acc_num/data_num

def rouge(output_ids, ans, ignore_len):
    set_ans = set(ans)
    set_ans.remove(128000)
    output = output_ids[ignore_len:].tolist()
    set_out = set(output)
    
    score =  set_out & set_ans 
    return [len(score),len(set_ans)]

ds = load_dataset("rajpurkar/squad")
device='cuda'
u=int(sys.argv[1])

model_normal = AutoModelForCausalLM.from_pretrained("./model/testgr"+str(u+1))
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

data_size_v = 400

validation_dataset=ds["validation"].shuffle(seed=1)

data_v = make_data(validation_dataset)
size_v = int(len(data_v)/4)

input_ids_tensor_v = make_tensor(data_v, "input_ids", size_v)
labels_tensor_v = make_tensor(data_v, "labels", size_v)
attention_mask_tensor_v = make_tensor(data_v, "attention_mask", size_v)
ans = [data["ans"] for data in data_v]


vocab_size = model_normal.config.vocab_size
criterion = torch.nn.CrossEntropyLoss(ignore_index=128001)

criterion.to(device)

input_ids_tensor_v=input_ids_tensor_v.to(device)
labels_tensor_v=labels_tensor_v.to(device)
attention_mask_tensor_v=attention_mask_tensor_v.to(device)

model_normal.to(device)
# model_train.to(device)
# model_distill.to(device)

model_normal.eval()
# model_train.eval()
# model_distill.eval()

losses=0
acc=0
words_len=0

for i in range(size_v):
    
    input_ids_v=input_ids_tensor_v[i]
    labels_v=labels_tensor_v[i]
    attention_mask_v=attention_mask_tensor_v[i]
    with torch.no_grad():
        output_ids = model_normal.generate(input_ids=input_ids_v, attention_mask=attention_mask_v, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
    
    

    for j in range(4):
        rgh = rouge(output_ids[j], ans[i*4+j], ignore_len=256)
        acc += rgh[0]
        words_len += rgh[1]
        seq_len=len(ans[i*4+j])-1

print(f"acc: {(acc/words_len):.3f}")    
