from openai import OpenAI

if __name__ == '__main__':
    base_url = 'https://open.bigmodel.cn/api/paas/v4'
    api_key = '0ea786d9905743a4ac9a4cf7b9e74719.TDE1JQlvq9xVHdxf'
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    data = {
        'model': 'glm-4-plus',
        'messages': [
            {'role': 'system', 'content': '你是一个图书管理员'},
            {'role': 'assistant', 'content': '图书借阅和归还，根据喜好为读者推荐图书'},
            {'role': 'user', 'content': '我想借一本关于人工智能的书'},
        ],
        'stream': False
    }
    response = client.chat.completions.create(**data)
    print(response.choices[0].message.content)
