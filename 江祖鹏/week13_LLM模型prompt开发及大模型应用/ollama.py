import requests
import json

if __name__ == "__main__":
    
    url  = "http://localhost:11434/api/generate"

    #提供JSON格式调用
    data = {
        "model" : "deepseek-r1:1.5b",
        "prompt" : "你好",
        "stream" : False
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print(json.loads(response.text)["response"])
    else:
        print(f'加载失败，状态码：{response.status_code}')


