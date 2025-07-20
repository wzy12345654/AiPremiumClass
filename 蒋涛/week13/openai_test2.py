"""
2. 利用OpenAI API 调用远端大模型API，调试参数观察输出结果的差异。
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
    response = client.chat.completions.create(
        model="glm-4-flashx",
        messages=[
            {"role": "system", "content": "输出的内容需要用json格式，key为event，value为事件的详细描述，包括时间、地点、人物、事件的详细描述。描述的事件要有相关性"},
            {"role": "user", "content": "今天是6月19日，请列举历史上的今天发生过的十件大事件。"}
        ],
        # temperature=1,
        top_p=0.5,
        max_tokens=500,
    )
    print(response.choices[0].message.content)
