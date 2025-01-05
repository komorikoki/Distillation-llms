from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(
    "../SQuAD/model/distill_model",
    quantization_config=quantization_config
)

from peft import get_peft_model, LoraConfig, TaskType
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],
    inference_mode=False,
    r=8,  # 低ランクアダプターのランク
    lora_alpha=32,  # アダプターのスケール
    lora_dropout=0.1,  # ドロップアウト率
)

model = get_peft_model(model, lora_config)

model.save_pretrained("./model/QLoRA_distill_model")