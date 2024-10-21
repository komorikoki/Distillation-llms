from transformers import AutoTokenizer 
import transformers
import torch

model = "mistral-merged"
messages = [{"role": "user", "content": "What is your name ?"}]
chat_template = """
{% if messages | length > 0 %}
    User: {{ messages[0].content }}
{% endif %}
{% if messages | length > 1 %}
    Assistant: {{ messages[1].content }}
{% endif %}
"""
tokenizer = AutoTokenizer.from_pretrained(model)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template = chat_template)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])