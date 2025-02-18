from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import re
from tqdm import tqdm
import torch
from torch.optim import AdamW
import matplotlib.pyplot as plt
from torch import nn

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
device='cuda'
# モデルの準備
model_before = AutoModelForCausalLM.from_pretrained("./model/normal_model")
model_after = AutoModelForCausalLM.from_pretrained("./model/distill_model")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

data_size = 800
size = int(data_size/4)

validation_dataset=ds["test"].shuffle(seed=42)

def reshape(dataset):
    dataset=dataset["text"]
    dataset = [item for item in dataset if item != '' and len(item) >= 50 and '@' not in item]
    dataset = [re.sub(r'[^a-zA-Z0-9 .?]', '', item) for item in dataset]
    dataset = [re.sub(r'\s+', ' ', item) for item in dataset]
    print(len(dataset))
    return dataset[:data_size]

def max_length(dataset):
    max_eval=0
    for i in dataset:
        max_eval = len(i) if len(i) > max_eval else max_eval
    print(max_eval)
    return

def calc_length(arr):
    num=0
    for data in arr:
        num += 1
        if data == tokenizer.pad_token_id:
            break
    return num-1

dataset=reshape(validation_dataset)
max_length(dataset)

def batch(input):
    batch_train=[]
    for i in range(size):
        batch_input=[input[4*i+0], input[4*i+1], input[4*i+2], input[4*i+3]]
        batch_train.append(batch_input)

    return batch_train

def accuracy(top_preds, labels, ignore_index):
    data_num=0
    acc_num=0
    for i in range(labels.size(0)):
        if labels[i]!=ignore_index:
            data_num += 1
            if torch.any(top_preds[i]==labels[i]):
                acc_num += 1   
    return acc_num/data_num



# 入力とラベルを設定
data = []
for text in tqdm(dataset, desc="Tokenizing dataset"):
    tokenized = tokenizer(text, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    input_ids = tokenized['input_ids'].squeeze().tolist()
    attention_mask = tokenized['attention_mask'].squeeze().tolist()
    labels = input_ids[1:] + [tokenizer.pad_token_id]
    data.append({"input_ids": input_ids, "labels": labels, "attention_mask":attention_mask})


input_ids = [item["input_ids"] for item in data]
labels = [item["labels"] for item in data]
attention_mask = [item["attention_mask"] for item in data]

input_ids = batch(input_ids)
labels = batch(labels)
attention_mask = batch(attention_mask)

input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
labels_tensor = torch.tensor(labels, dtype=torch.long)
attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)


# 仮定: ボキャブラリサイズと頻出語のトークンIDを定義
vocab_size = model_after.config.vocab_size

input_ids_tensor=input_ids_tensor.to(device)
labels_tensor=labels_tensor.to(device)
attention_mask_tensor = attention_mask_tensor.to(device)
model_before.to(device)
model_after.to(device)

model_before.eval()
model_after.eval()

i=1
rank=5

acc_before = 0
acc_after = 0
for i in tqdm(range(size)):
    input_ids=input_ids_tensor[i] 
    labels=labels_tensor[i]
    attention_mask=attention_mask_tensor[i]
    with torch.no_grad():
        outputs_before = model_before(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs_after = model_after(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits_before = outputs_before.logits
        logits_after = outputs_after.logits

    top_preds_before = torch.topk(logits_before, k=rank, dim=-1).indices
    top_preds_after = torch.topk(logits_after, k=rank, dim=-1).indices
    acc_before += accuracy(top_preds_before.view(-1, rank), labels.view(-1), tokenizer.pad_token_id)
    acc_after += accuracy(top_preds_after.view(-1, rank), labels.view(-1), tokenizer.pad_token_id)

print(f"acc: {(acc_before/size):.3f}")
print(f"acc: {(acc_after/size):.3f}")


