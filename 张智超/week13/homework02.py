import os
from openai import OpenAI
from dotenv import find_dotenv, load_dotenv

# 加载.env中的内容到系统变量
load_dotenv(find_dotenv())

client = OpenAI(
    api_key = os.getenv('API_KEY'),
    base_url = os.getenv('BASE_URL')
)

def do_request(top_p = 0.9):
    resp = client.chat.completions.create(
        model = 'glm-4-flash-250414',
        messages= [
            {'role': 'user', 'content': '请推荐3本书籍，内容控制在100字以内。'}
        ],
        top_p = top_p
    )
    print(f'=========top_p={top_p}')
    print(resp.choices[0].message.content)

if __name__ == '__main__':
    # 部分描述有差别
    for i in range(3):
        do_request()
    # 内容完全一样
    for i in range(3):
        do_request(0.1)
    
    
    