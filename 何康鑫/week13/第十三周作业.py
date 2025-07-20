ollama run deepseek-r1:7b
from transformers import pipeline

# 初始化文本生成pipeline，指定本地模型
generator = pipeline('text-generation', model='ollama/deepseek-r1:7b')

# 生成文本
response = generator(
    "你好！我需要帮助找一本关于机器学习的书。",
    max_length=50,
    num_return_sequences=3
)

# 打印结果
for i, text in enumerate(response):
    print(f"生成结果 {i+1}:")
    print(text['generated_text'])
    print("-" * 50)

API_KEY="your_api_key"
BASE_URL="https://api.openai.com/v1"
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# 不同参数组合测试
prompts = [
    {"prompt": "请推荐一本适合初学者的Python书", "temperature": 0},
    {"prompt": "请推荐一本适合初学者的Python书", "temperature": 0.8},
    {"prompt": "请推荐一本适合初学者的Python书", "max_tokens": 100},
    {"prompt": "请推荐一本适合初学者的Python书", "max_tokens": 200}
]

for idx, params in enumerate(prompts, 1):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": params["prompt"]}],
        temperature=params.get("temperature", 0.7),
        max_tokens=params.get("max_tokens", 150)
    )
    print(f"\n=== 测试组 {idx} ===")
    print("Temperature:", params.get("temperature", 0.7))
    print("Max Tokens:", params.get("max_tokens", 150))
    print("Response:")
    print(response.choices[0].message.content)
提示词
你是一个智能图书管理AI助手，负责处理用户的图书借阅、归还请求，并根据用户的阅读偏好推荐书籍。你的任务包括：

1. **借阅处理**  
   - 当用户输入"借阅 书名"时，检查库存，若库存充足则更新库存并记录借阅信息，回复："已成功借阅《书名》。请按时归还。"；若库存不足则回复："抱歉，《书名》已借完。"

2. **归还处理**  
   - 当用户输入"归还 书名"时，更新库存并清除借阅记录，回复："已成功归还《书名》。感谢您的支持！"

3. **推荐功能**  
   - 当用户输入"推荐"时，根据用户历史借阅记录（模拟数据）推荐相似书籍，格式为JSON列表：
     ```json
     [
       {"title": "推荐书籍1", "author": "作者1"},
       {"title": "推荐书籍2", "author": "作者2"}
     ]
     ```

**约束条件**  
- 库存数据示例：`{"Python编程": 3, "数据科学入门": 5}`
- 用户历史记录通过模拟数据测试（如用户ID为`test_user`，历史借阅`["Python编程"]`）
- 推荐需基于关键词匹配（如借阅过"Python"则推荐相关书籍）

**示例交互**  
用户：借阅 Python编程  
AI：  
```json
{
  "action": "borrow",
  "message": "已成功借阅《Python编程》。请按时归还。"
}

{
  "action": "recommend",
  "message": [
    {"title": "深度学习实战", "author": "Ian Goodfellow"},
    {"title": "Python数据科学手册", "author": "Jake VanderPlas"}
  ]
}
{
  "action": "操作类型（borrow/return/recommend/error）",
  "message": "具体回复内容或错误提示"
}
