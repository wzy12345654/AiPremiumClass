"""
3. 利用大模型提示词设计一个智能图书管理AI。
功能:实现图书借阅和归还。根据喜好为读者推荐图书。
"""
import os
from openai import OpenAI
from dotenv import find_dotenv, load_dotenv

if __name__=='__main__':
    load_dotenv(find_dotenv())
    client = OpenAI(
        api_key=os.environ['API_KEY'], # 必须指定，⽤于⾝份验证
        base_url=os.environ['BASE_URL'], # 提供模型调⽤的服务URL
    )
    system_content = """
    你是一个专业的图书管理AI，具备以下能力：
        1. 借阅管理：处理《书名》的借阅/归还请求，返回[成功/失败]及原因
        2. 推荐逻辑：根据用户历史借阅记录推荐3本书，格式为JSON：{"recommendations": [{"title":"书名", "reason":"推荐原因"}]}
        3. 回答规则：不回答与图书无关问题，用表格展示借阅记录

        用户历史借阅：《三体》《平凡的世界》
        当前请求：{}
        """
    response = client.chat.completions.create(
        model="glm-4-flashx",
        messages=[
            {"role": "system", "content": f"{system_content}"},
            {"role": "user", "content": "我来还《三国演义》，请给我推荐类似的书籍"}
        ],
        # temperature=1,
        top_p=0.5,
        max_tokens=200,
    )
    print(response.choices[0].message.content)
    
