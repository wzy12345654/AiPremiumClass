from ollama import chat
response = chat(
    model="qwen3:0.6b",
    messages=[
        {"role": "user", "content": "为什么天空是蓝色的？"}
    ]
)
print(response.message.content)