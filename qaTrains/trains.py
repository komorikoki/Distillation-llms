from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import re
from tqdm import tqdm
import torch
from torch.optim import AdamW
import matplotlib.pyplot as plt
from torch import nn

ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
device='cuda'
# モデルの準備
teacher_model = AutoModelForCausalLM.from_pretrained("../newTrains/finetuned_model5")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
size=2500
train_dataset=ds["train"].shuffle(seed=42).select(range(size*4))
validation_dataset=ds["validation"].shuffle(seed=42)

train_datas = []

for i in tqdm(range(size*4)):
    # 質問と答えを辞書形式で保存
    train_data = {
        'question': train_dataset['question'][i],
        'answer': train_dataset['answer'][i]['aliases'][0]
    }
    train_datas.append(train_data)

# アルファベット以外の文字を除去する関数
def clean_text(text):
    return re.sub(r'[^a-zA-Z ?]', '', text)

# 質問と回答を清掃
for data in train_datas:
    data['question'] = clean_text(data['question'])
    data['answer'] = clean_text(data['answer'])

train_data = []
for text in tqdm(train_datas, desc="Tokenizing dataset"):
    tokenized = tokenizer("Q: "+ text['question'] + " A:" + text['answer']+".", padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    input_ids = tokenized['input_ids'].squeeze().tolist()
    attention_mask = tokenized['attention_mask'].squeeze().tolist()
    quest_pos = len(tokenizer(text['question'])['input_ids'])
    labels = input_ids[1:] + [tokenizer.pad_token_id]
    for i in range(len(attention_mask)):
        if attention_mask[i]==0:
            labels[i]=-100
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



# クロスエントロピー損失関数の設定
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
criterion.to(device)

optimizer = AdamW(teacher_model.parameters(), lr=5e-5)
input_ids_tensor=input_ids_tensor.to(device)
labels_tensor=labels_tensor.to(device)
attention_mask_tensor = attention_mask_tensor.to(device)
teacher_model.to(device)
eval_loss=0
teacher_model.train()

eval_losses = []
vocab_size = teacher_model.config.vocab_size

for j in range(3):
    for i in tqdm(range(size)):
     
        input_ids=input_ids_tensor[i]
        labels=labels_tensor[i]
        attention_mask=attention_mask_tensor[i]
        optimizer.zero_grad()
        outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits=outputs.logits
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        eval_loss+= loss

        loss.backward()
        optimizer.step()
        if i%100==99:
            eval_loss /= 100
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
plt.savefig('eval_loss_plot.png')

teacher_model.save_pretrained("./instruction_model2")
