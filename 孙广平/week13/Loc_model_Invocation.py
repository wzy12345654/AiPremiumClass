import requests
import json

if __name__ == "__main__":
    url = "http://localhost:11434/api/generate"

    data = {
        "model": "deepseek-r1:1.5b",
        "prompt": "世界上最高的十座山峰是?",
        "stream": False,
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()
        print(result["response"])
    else:
        print("请求失败:", response.status_code)
