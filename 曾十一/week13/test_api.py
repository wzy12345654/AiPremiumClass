import requests

# API 地址（以 OpenAI 为例）
url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# 请求头（包括 API 密钥）
headers = {
    "Authorization": "e5bdd8d6aedc4f319855cf41bc2e85f1.caOMjjtZTMdDAkkh",
    "Content-Type": "application/json"
}

# 请求体
data = {
    "model": "glm-4-flash",
    "messages": [{"role": "user", "content": "你好，介绍一下量子计算"}],
    "temperature": 0.7
}

# 发送 POST 请求
response = requests.post(url, headers=headers, json=data)

# 处理响应
if response.status_code == 200:
    result = response.json()
    print(result["choices"][0]["message"]["content"])
else:
    print(f"请求失败，状态码：{response.status_code}")
    print(response.text)
