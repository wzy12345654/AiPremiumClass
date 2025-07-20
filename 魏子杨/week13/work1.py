import requests
import json
if __name__ == '__main__':
 # url = 'http://localhost:11434/api/chat'
 url = 'http://localhost:11434/api/generate'
 data = {
    "model": "qwen3:0.6b",
    "prompt": "⼤海为什么是蓝⾊的？",
    "stream": False  #不要一个一个字输出
 }
 response = requests.post(url, json=data)
print(response.text)