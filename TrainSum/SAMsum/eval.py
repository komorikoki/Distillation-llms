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
    cq = text[:(text.find("S:") + 3)]
    ans = text[(text.find("S:") + 3):]
    return [cq, ans]

def make_data(data):
    dataset=reshape(data)
    data = []
    for text in tqdm(dataset, desc="Tokenizing dataset"):
        [c, ans] = devide(text)
        tokenized = tokenizer(c, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
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


ds = load_dataset("knkarthick/samsum")
device='cuda'
model_normal = AutoModelForCausalLM.from_pretrained("./model/train_model_QLoRA5")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

data_size_v = 100
size_v = int(data_size_v/4)
validation_dataset=ds["validation"].shuffle(seed=42)

data_v = make_data(validation_dataset)

input_ids_tensor_v = make_tensor(data_v, "input_ids", size_v)
labels_tensor_v = make_tensor(data_v, "labels", size_v)
attention_mask_tensor_v = make_tensor(data_v, "attention_mask", size_v)
ans = [data["ans"] for data in data_v]

def rouge(output_ids, ans, ignore_len):
    output = output_ids[ignore_len:].tolist()
    set_out = set(output)
    set_ans = set(ans)
    score =  set_out & set_ans
    print(score)   
    return len(score)/len(ans)

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
rge=0

with open("output.txt", "w") as f:
    for i in tqdm(range(size_v)):
        
        input_ids_v=input_ids_tensor_v[i]
        labels_v=labels_tensor_v[i]
        attention_mask_v=attention_mask_tensor_v[i]
        with torch.no_grad():
            output_ids = model_normal.generate(input_ids=input_ids_v, attention_mask=attention_mask_v, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
        
        

        for j in range(4):
            rge += rouge(output_ids[j], ans[i*4+j], ignore_len=512)
            seq_len=len(ans[i*4+j])-1
            f.write(str(4*i+j) + " label:" + tokenizer.decode(input_ids_v[j][500:511]) +"\n")
            f.write(str(4*i+j) + " ans:" + tokenizer.decode(ans[i*4+j][1:]) +"\n")
            f.write(str(4*i+j) + " model:" + tokenizer.decode(output_ids[j][512:512+seq_len]) +"\n")

print(f"rge: {(rge/data_size_v):.3f}")    
