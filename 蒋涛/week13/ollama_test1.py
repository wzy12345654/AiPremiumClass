import requests
import json

if __name__ == '__main__':
    url = 'http://localhost:11434/api/generate'
    data = {
        'model': 'deepseek-r1:7b',
        'prompt': '你是一个图书管理员，用简体中文回答.<think></think>',
        'stream': False
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print(response.text)
        print(json.loads(response.text))
    else:
        print(f"请求失败，状态码: {response.status_code}")