from transformers import AutoTokenizer, pipeline

# Hugging Faceのテキスト生成パイプラインを作成
generator = pipeline("text-generation", model="mistral-merged", device="cuda")
tokenizer = AutoTokenizer.from_pretrained("mistral-merged")

while True:
    # ユーザー入力を受け取る
    user_input = input("User: ")

    # 終了条件
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat.")
        break

    # モデルにプロンプトを渡して出力を得る
    prompt = f"User: {user_input}\nAI:"
    outputs = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)

    # モデルの出力を表示
    print(f"AI: {outputs[0]['generated_text'].split('AI:')[-1]}")