import requests

# API 地址（智谱大模型）
url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# 请求头（注意加上 Bearer）
headers = {
    "Authorization": "Bearer e5bdd8d6aedc4f319855cf41bc2e85f1.caOMjjtZTMdDAkkh",
    "Content-Type": "application/json"
}

# 多轮对话内容，包括系统角色定义 + 用户请求
data = {
    "model": "glm-4-flash",
    "messages": [
        {
            "role": "system",
            "content": (
                "你是一个智能图书管理员，支持图书借阅、归还和个性化推荐。\n"
                "功能说明：\n"
                "1. 用户说“我想借xxx”，你记录借书信息并回复。\n"
                "2. 用户说“我还了xxx”，你更新借阅状态。\n"
                "3. 用户说“我喜欢xx类型的书”，你推荐对应图书。\n"
                "4. 推荐时避免用户已借过的图书，回复尽量自然、亲切。"
            )
        },
        {"role": "user", "content": "我想借《三体》"},
        #{"role": "user", "content": "我喜欢悬疑小说，有什么推荐？"},
        #{"role": "user", "content": "我还了《三体》"}
    ],
    "temperature": 0.7,
    "stream": False
}

# 发送 POST 请求
response = requests.post(url, headers=headers, json=data)

# 处理响应
if response.status_code == 200:
    result = response.json()
    print(result["choices"][0]['message']["content"])
else:
    print(f"请求失败，状态码：{response.status_code}")
    print(response.text)
