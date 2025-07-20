#安装ollama，下载模型并用代码方式调用。
import requests
import json

if __name__ == "__main__":
    url = "http://localhost:11434/api/generate"
    data = {
        "model":"qwen3:0.6b",
        "prompt": "梅雨季节的产生原因？",
        "stream": False,
    }
    response= requests.post(url, json=data)
    if response.status_code == 200:  # 请求成功
        print(json.loads(response.text)['response'])
