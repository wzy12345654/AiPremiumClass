import requests
import json

if __name__ == "__main__":
    url = "http://localhost:11434/api/generate"

    # 提供json格式调用数据
    data = {
        "model":"llama3.2",
        "prompt": "喜欢和爱的区别是什么？/nothink", # /nothink表示不使用思考链
        "stream": False,  # 是否开启流式输出
    }

    response= requests.post(url, json=data)

    if response.status_code == 200:  # 请求成功
        print(json.loads(response.text)['response'])