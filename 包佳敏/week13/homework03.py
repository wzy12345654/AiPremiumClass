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
            {'role':'system', 'content':'你是一个智能图书管理AI'},
            {'role':'user', 'content':'实现以下几个功能：'
            '1、图书的借阅：提示用户如何操作，操作需扫图书二维码借阅'
            '  （a)、提示用户如何操作，操作需扫图书二维码借阅'
            '  （b）、显示图书封面、名字、作者、出版年份等'
            '2、图书的归还'
            '  （a)、提示用户如何操作，操作需扫图书二维码归还'
            '  （b）、把扫码书籍放置还书架'
            '3、为读者推荐图书：根据用户的年龄、性别、历史借阅记录分析、当下热门书籍等分析等为读者推荐书籍'
            '  （a)、推荐结果分页显示，每页是10个'
            '  （b）、显示图书封面、名字、作者、出版年份等'
            },
            {'role':'user', 'content':'你好，我需要还书'}
        ],
        # 模型参数
        temperature=0.95,
        # 最大token数
        max_tokens=500,
    )

    # 打印结果
    print(response.choices[0].message.content)