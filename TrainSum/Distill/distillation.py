from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import re
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.optim import AdamW
import matplotlib.pyplot as plt
from torch import nn

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
device='cuda'
# モデルの準備
teacher_model = AutoModelForCausalLM.from_pretrained("./model/teacher_model1")
model = AutoModelForCausalLM.from_pretrained("./model/initialized_distill_model2")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

data_size = 100000
size = int(data_size/4)

train_dataset=ds["train"].shuffle(seed=42).select(range(500000))

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


dataset=reshape(train_dataset)
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
vocab_size = model.config.vocab_size

# クロスエントロピー損失関数の設定
criterion = torch.nn.CrossEntropyLoss(ignore_index=128001)
criterion.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
input_ids_tensor=input_ids_tensor.to(device)
labels_tensor=labels_tensor.to(device)
attention_mask_tensor = attention_mask_tensor.to(device)
teacher_model.to(device)
model.to(device)
model.train()
criterion.to(device)
alpha=0.5
temperature=1.0
epochs = 1

for j in range(epochs):
    for i in tqdm(range(size)):
        
        input_ids=input_ids_tensor[i]
        labels=labels_tensor[i]
        attention_mask=attention_mask_tensor[i]
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits=outputs.logits
        student_prob=F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            teacher_outputs_logits=teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
            teacher_prob=F.softmax(teacher_outputs_logits, dim=-1)
        
        cr_loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        kldiv_loss=F.kl_div(student_prob, teacher_prob, reduction="none")
        kl_div_per_token = kldiv_loss.sum(dim=-1)
        kl_loss=(kl_div_per_token * attention_mask).sum()/attention_mask.sum()
        
        loss = alpha * cr_loss + (1-alpha) * kl_loss
        loss.backward()
        optimizer.step()
        
    print("done: ", j+1, "/", epochs)



model.save_pretrained("./model/distill2_teacher1distill")



