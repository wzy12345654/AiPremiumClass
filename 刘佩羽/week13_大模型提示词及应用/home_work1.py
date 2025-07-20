import requests
import json

''' 本地ollama模型调用，模式：chat '''
if __name__ == '__main__':
    url = 'http://localhost:11434/api/chat'
    # 提供数据给ollama模型，来调用
    data = {
        "model": "llama3.2:latest",
        "messages": [
            {
                "role": "user",
                "content": "请跟我打个招呼"
            }
        ],
        "stream": False
    }
    # 发送post请求
    response = requests.post(url, json=data)

    if response.status_code == 200:
        response = json.loads(response.text)
        print(response['message']['content'])
    else:
        print('请求失败')

''' 本地ollama模型调用，模式：generate 
import requests
import json

if __name__ == '__main__':
    
    url = 'http://localhost:11434/api/generate'
    # 提供数据给ollama模型，来调用
    data = {
        "model": "llama3.2:latest",
        "prompt": "你好，介绍一下你自己",
        "system": "你是一个中文助手,使用中文作为母语进行回复",
        "stream": True
    }
    # 发送post请求，启用流式响应
    response = requests.post(url, json=data, stream=True)

    if response.status_code == 200:
        # 初始化缓冲区
        buffer = ''
        # 逐块读取响应内容
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                # 将缓冲区内容添加到块中
                chunk = buffer + chunk.decode('utf-8')
                # 初始化一个空列表来存储解析的 JSON 对象
                json_objects = []
                while chunk:
                    try:
                        # 尝试解析 JSON 对象
                        obj, pos = json.JSONDecoder().raw_decode(chunk)
                        json_objects.append(obj)
                        # 移除已解析的部分
                        chunk = chunk[pos:].lstrip()
                    except json.JSONDecodeError:
                        # 如果解析失败，将剩余的部分添加到缓冲区
                        buffer = chunk
                        break
                # 打印解析的 JSON 对象
                for obj in json_objects:
                    print(obj['response'], end='')
    else:
        print('请求失败')
'''
    