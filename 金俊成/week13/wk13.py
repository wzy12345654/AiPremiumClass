
"""
3. 功能说明
功能
描述
图书借阅
输入 借阅 <书名> 即可尝试借阅指定书籍。
图书归还
输入 归还 <书名> 可将书籍归还至图书馆。
推荐图书
输入 推荐 将根据用户A的阅读偏好推荐相关书籍。
自然对话
支持自由提问，例如“今天有什么新书？”、“如何借书？”等常见问题。
"""
from openai import OpenAI

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-jfxgwhgqleoltzwysdnkdhkflonoystyvaelmgaphmjbfvzg",
    base_url="https://api.siliconflow.cn/v1"
)

# 模拟图书数据库
books_db = {
    "百科全书": {"available": True, "category": "科普"},
    "机器学习基础": {"available": True, "category": "技术"},
    "Python编程入门": {"available": True, "category": "编程"},
    "人工智能导论": {"available": True, "category": "技术"}
}

# 用户阅读偏好模拟
user_preferences = {
    "用户A": ["技术", "编程"]
}

def update_book_status(title, status):
    """更新书籍状态"""
    if title in books_db:
        books_db[title]["available"] = status
        return f"《{title}》已{'归还' if status else '借出'}。"
    else:
        return "未找到该书。"

def recommend_books(user):
    """根据用户偏好推荐图书"""
    if user in user_preferences:
        categories = user_preferences[user]
        recommended = [title for title, info in books_db.items() if info["category"] in categories and info["available"]]
        return "为您推荐：" + "、".join(recommended) if recommended else "暂无推荐书籍。"
    else:
        return "无法获取您的偏好，请先登录或提供阅读兴趣。"

# 构建初始系统消息
system_prompt = """
你是一个智能图书管理员AI，具备以下功能：
1. 图书借阅：检查书籍库存{{books_db}}并完成借阅流程。
2. 图书归还：接收用户的归还请求，更新书籍状态为可借阅。
3. 个性化推荐：根据用户的阅读历史和兴趣偏好，推荐相关书籍。
4. 交互友好性：保持自然对话风格，理解用户意图并提供清晰反馈。
"""

messages = [
    {'role': 'system', 'content': system_prompt},
    {'role': 'assistant', 'content': '您好！我是智能图书管理员，可以帮助您借阅、归还书籍以及推荐适合您的读物。'}
]

# 主循环处理用户输入
while True:
    print(f'目前可借阅书籍：{[k for k,v in books_db.items() if v['available']]}')
    user_input = input("请输入您的指令（如借阅/归还/推荐）：")

    if user_input.lower() == '退出':
        print("感谢使用智能图书管理系统，再见！")
        break

    # 处理图书借阅
    elif user_input.startswith("借阅"):
        book_title = user_input[2:].strip()
        if book_title in books_db and books_db[book_title]["available"]:
            messages.append({'role': 'user', 'content': f"我想借阅《{book_title}》。"})
            response = client.chat.completions.create(
                model="Qwen/Qwen3-8B",
                messages=messages,
                stream=True
            )
            for chunk in response:
                if not chunk.choices:
                    continue
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()
            # 更新书籍状态
            update_book_status(book_title, False)
        else:
            print(f"抱歉，《{book_title}》目前不可借阅。")

    # 处理图书归还
    elif user_input.startswith("归还"):
        book_title = user_input[2:].strip()
        result = update_book_status(book_title, True)
        print(result)

    # 处理推荐请求
    elif user_input.startswith("推荐"):
        recommendations = recommend_books("用户A")
        print(recommendations)

    # 默认情况：普通聊天
    else:
        messages.append({'role': 'user', 'content': user_input})
        response = client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=messages,
            stream=True
        )
        full_response = ""
        for chunk in response:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                print(content, end="", flush=True)
        # 将助手的回答加入消息历史
        messages.append({'role': 'assistant', 'content': full_response})

print("\n对话结束。")
