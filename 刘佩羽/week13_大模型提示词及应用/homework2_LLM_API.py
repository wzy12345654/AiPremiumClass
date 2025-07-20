from openai import OpenAI
from dotenv import load_dotenv
import os

# load environment variables from .env file
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_URL")
)

# 添加一个开关以允许流式输出
stream = True

if stream:
    completion = client.chat.completions.create(
        model="glm-4",  
        messages=[    
            {"role": "system", "content": "你是一个聪明且富有创造力的小说作家"},    
            {"role": "user", "content": "请你作为童话故事大王，写一篇短篇童话故事，故事的主题是要永远保持一颗善良的心，要能够激发儿童的学习兴趣和想象力，同时也能够帮助儿童更好地理解和接受故事中所蕴含的道理和价值观。"} 
        ],
        top_p=0.7,
        temperature=0.9,
        stream=True  # 启用流式输出
    )

    # 迭代流式响应
    for chunk in completion:
        print(chunk.choices[0].delta.content, end='', flush=True)
else:
    completion = client.chat.completions.create(
        model="glm-4",  
        messages=[    
            {"role": "system", "content": "你是一个聪明且富有创造力的小说作家"},    
            {"role": "user", "content": "请你作为童话故事大王，写一篇短篇童话故事，故事的主题是要永远保持一颗善良的心，要能够激发儿童的学习兴趣和想象力，同时也能够帮助儿童更好地理解和接受故事中所蕴含的道理和价值观。"} 
        ],
        top_p=0.7,
        temperature=0.9
    )

    print(completion.choices[0].message)
