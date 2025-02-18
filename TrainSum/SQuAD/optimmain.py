import os
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer

print("start")
student_model = AutoModelForCausalLM.from_pretrained("../Distill/model/distill_model")
student_model.save_pretrained("./model/testpp0")

del student_model

for i in range(10):
    os.system(f"python optimeval.py {i}")
    os.system(f"python optimdistill.py {i}")
    # os.system(f"python optimacc.py {i}")
