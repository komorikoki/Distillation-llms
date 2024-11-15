from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import re
from tqdm import tqdm
import torch
from torch.optim import AdamW
import matplotlib.pyplot as plt

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
device='cuda'
# モデルの準備
teacher_model = AutoModelForCausalLM.from_pretrained("../distillLLAMA2")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

train_dataset=ds["train"].shuffle(seed=42).select(range(500000))
validation_dataset=ds["validation"].shuffle(seed=42)
train_dataset = train_dataset["text"]
train_dataset = [item for item in train_dataset if item != '' and len(item) >= 50 and '@' not in item]
validation_dataset=validation_dataset["text"]
validation_dataset = [item for item in validation_dataset if item != '' and len(item) >= 50 and '@' not in item]



train_dataset = [re.sub(r'[^a-zA-Z ]', '', item) for item in train_dataset]
train_dataset = [re.sub(r'\s+', ' ', item) for item in train_dataset]
validation_dataset = [re.sub(r'[^a-zA-Z ]', '', item) for item in validation_dataset]
validation_dataset = [re.sub(r'\s+', ' ', item) for item in validation_dataset]

size=25000


# 入力とラベルを設定
train_data = []
for text in tqdm(train_dataset, desc="Tokenizing dataset"):
    tokenized = tokenizer(text, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    input_ids = tokenized['input_ids'].squeeze().tolist()
    attention_mask = tokenized['attention_mask'].squeeze().tolist()
    labels = input_ids[1:] + [tokenizer.pad_token_id]
    labels[-1]=-100
    train_data.append({"input_ids": input_ids, "labels": labels, "attention_mask":attention_mask})


input_ids = [item["input_ids"] for item in train_data]
labels = [item["labels"] for item in train_data]
attention_mask = [item["attention_mask"] for item in train_data]

batch_train=[]
for i in range(size):
    batch_input=[input_ids[4*i+0], input_ids[4*i+1], input_ids[4*i+2], input_ids[4*i+3]]
    batch_train.append(batch_input)
input_ids=batch_train

batch_train=[]
for i in range(size):
    batch_input=[labels[4*i+0], labels[4*i+1], labels[4*i+2], labels[4*i+3]]
    batch_train.append(batch_input)
labels=batch_train

batch_train=[]
for i in range(size):
    batch_input=[attention_mask[4*i+0], attention_mask[4*i+1], attention_mask[4*i+2], attention_mask[4*i+3]]
    batch_train.append(batch_input)
attention_mask=batch_train

input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
labels_tensor = torch.tensor(labels, dtype=torch.long)
attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)


# 仮定: ボキャブラリサイズと頻出語のトークンIDを定義
vocab_size = teacher_model.config.vocab_size
weights = torch.ones(vocab_size)  # 全単語の重みを 1 に初期化
for i in input_ids:
    for j in i:
        weights[j]+=0

weights=1/weights

# クロスエントロピー損失関数の設定
criterion = torch.nn.CrossEntropyLoss(weight=weights)
criterion.to(device)

optimizer = AdamW(teacher_model.parameters(), lr=5e-5)
input_ids_tensor=input_ids_tensor.to(device)
labels_tensor=labels_tensor.to(device)
attention_mask_tensor = attention_mask_tensor.to(device)
teacher_model.to(device)
eval_loss=0
teacher_model.train()

eval_losses = []

for j in range(3):
    for i in tqdm(range(size)):
     
        input_ids=input_ids_tensor[i]
        labels=labels_tensor[i]
        attention_mask=attention_mask_tensor[i]
        optimizer.zero_grad()
        outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

        eval_loss+= loss

        loss.backward()
        optimizer.step()
        if i%1000==999:
            eval_loss /= 1000
            print("eval_loss", i, ":", eval_loss)
            eval_losses.append(eval_loss.item())  # 100ステップごとにeval_lossを保存
            eval_loss=0
    print("done:", j, "eval_loss:", eval_loss)
    eval_loss=0

# プロット
print(eval_losses)
plt.plot(eval_losses)
plt.xlabel('Steps (×100)')
plt.ylabel('Loss')
plt.title('Evaluation Loss Over Time')
plt.savefig('eval_loss_plot11.png')

teacher_model.save_pretrained("./finetuned_model4")
