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
model_before = AutoModelForCausalLM.from_pretrained("./model/teacher_model1")
model_after = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

data_size = 766
size = int(data_size/4)

validation_dataset=ds["validation"].shuffle(seed=42)

def reshape(dataset):
    dataset=dataset["text"]
    dataset = [item for item in dataset if item != '' and len(item) >= 50 and '@' not in item]
    dataset = [re.sub(r'[^a-zA-Z0-9 ?]', '', item) for item in dataset]
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
vocab_size = model_before.config.vocab_size

# クロスエントロピー損失関数の設定
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
criterion.to(device)

input_ids_tensor=input_ids_tensor.to(device)
labels_tensor=labels_tensor.to(device)
attention_mask_tensor = attention_mask_tensor.to(device)
model_before.to(device)
model_after.to(device)

model_before.eval()
model_after.eval()

criterion.to(device)
eval_loss_before=0
eval_loss_after=0

with open("output.txt", "w") as f:
    for i in tqdm(range(size)):
        
        input_ids=input_ids_tensor[i] 
        labels=labels_tensor[i]
        attention_mask=attention_mask_tensor[i]
        with torch.no_grad():
            outputs_before = model_before(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs_after = model_after(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits_before = outputs_before.logits
            logits_after = outputs_after.logits
            prob_before=torch.argmax(logits_before, dim=-1)
            prob_after=torch.argmax(logits_after, dim=-1)
            
            
        for j in range(4):
            seq_len=calc_length(labels[j])
            f.write(str(4*i+j) + " label         :" + tokenizer.decode(labels[j][:seq_len]) +"\n")
            f.write(str(4*i+j) + " output_before :" + tokenizer.decode(prob_before[j][:seq_len]) +"\n")
            f.write(str(4*i+j) + " output_after :" + tokenizer.decode(prob_after[j][:seq_len]) +"\n\n")

                

        loss_before = criterion(logits_before.view(-1, vocab_size), labels.view(-1))
        loss_after = criterion(logits_after.view(-1, vocab_size), labels.view(-1))
        eval_loss_before += loss_before
        eval_loss_after += loss_after

print(f"loss_before {(eval_loss_before / data_size).item():.3f}")
print(f"loss_after {(eval_loss_after / data_size).item():.3f}")


