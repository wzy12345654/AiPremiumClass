import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

if __name__ == "__main__":
 
    load_dotenv(find_dotenv())
    
    # 创建调用客户端
    client = OpenAI(
        api_key=os.environ["api_key"],
        base_url=os.environ["base_url"]
    )

    # chat模式调用模型
    response = client.chat.completions.create(
        # 模型名称
        model="glm-4-flash-250414",
        # 消息列表
        messages=[  # 聊天历史信息
            {'role':'system', 'content':'你是一个暖男小弟弟形象的情场高手'},
            {'role':'user', 'content':'我叫蛋挞！'},
            {'role':'assistant', 'content':'你好蛋挞，Nice to meet you！'},
            {'role':'user', 'content':'我叫什么名字？初次沟通你对我是什么印象？'},
            {'role':'user', 'content':'你可以给我推荐好吃的蛋挞和蛋挞的做法吗'},
            {'role':'user', 'content':'吃多了蛋挞会胖吗？怎么办？'},
        ],
        # 模型参数
        temperature=0.95,
        # 最大token数
        max_tokens=500,
    )

    # 打印结果
    print(response.choices[0].message.content)