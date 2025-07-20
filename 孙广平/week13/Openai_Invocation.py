from openai import OpenAI

client = OpenAI(api_key="sk-***********9960c9e9d9142", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一位旅游顾问"},
        {"role": "user", "content": "推荐中国最美的十个地方"},
        {"role": "assistant", "content": "推荐如下：..."},
        {"role": "user", "content": "请说明推荐理由，不超过40字"},
    ],
    stream=False,
    max_tokens=1000,
    temperature=1
)

print(response.choices[0].message.content)
