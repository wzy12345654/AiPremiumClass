import requests
import json

def call_ollama_api():
    """
    调用Ollama API进行聊天对话（非流式）
    """
    # API端点
    url = "http://localhost:11434/api/chat"
    
    # 请求数据
    data = {
        "model": "qwen3",
        "messages": [
            {
                "role": "user",
                "content": "讲一个小笑话"
            }
        ],
        "stream": False
    }
    
    try:
        # 发送POST请求
        response = requests.post(url, json=data)
        
        # 检查响应状态
        if response.status_code == 200:
            # 解析JSON响应
            result = response.json()
            
            print("API调用成功! (非流式)")
            print("=" * 50)
            print("响应内容:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("=" * 50)
            
            # 提取并打印AI的回复
            if 'message' in result and 'content' in result['message']:
                print("\nAI回复:")
                print(result['message']['content'])
            
        else:
            print(f"API调用失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("连接错误: 无法连接到Ollama服务")
        print("请确保Ollama服务正在运行在 http://localhost:11434")
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"原始响应: {response.text}")

def call_ollama_api_stream():
    """
    调用Ollama API进行聊天对话（流式输出）
    """
    # API端点
    url = "http://localhost:11434/api/chat"
    
    # 请求数据
    data = {
        "model": "qwen3",
        "messages": [
            {
                "role": "user",
                "content": "讲一个小笑话"
            }
        ],
        "stream": True
    }
    
    try:
        # 发送POST请求，设置stream=True来获取流式响应
        response = requests.post(url, json=data, stream=True)
        
        # 检查响应状态
        if response.status_code == 200:
            print("API调用成功! (流式输出)")
            print("=" * 50)
            print("AI回复:")
            
            # 逐行读取流式响应
            for line in response.iter_lines():
                if line:
                    # 解码并解析每一行JSON
                    line_str = line.decode('utf-8')
                    
                    # 跳过最后一行（通常是done消息）
                    if line_str.strip() == '{"done":true}':
                        break
                    
                    try:
                        chunk = json.loads(line_str)
                        
                        # 提取并打印内容片段
                        if 'message' in chunk and 'content' in chunk['message']:
                            content = chunk['message']['content']
                            print(content, end='', flush=True)
                            
                    except json.JSONDecodeError:
                        # 忽略无法解析的行
                        continue
            
            print("\n" + "=" * 50)
            print("流式输出完成!")
            
        else:
            print(f"API调用失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("连接错误: 无法连接到Ollama服务")
        print("请确保Ollama服务正在运行在 http://localhost:11434")
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")

if __name__ == "__main__":
    # print("正在调用Ollama API...")
    # print("\n1. 非流式输出:")
    # call_ollama_api()
    
    # print("\n" + "="*80 + "\n")
    
    print("2. 流式输出:")
    call_ollama_api_stream()
