import os 
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

if __name__ == "__main__":

    # 加载.env文件
    load_dotenv(find_dotenv())

    #创建调用客户端
    client = OpenAI(
        api_key=os.environ['api_key'],
        base_url=os.environ['base_url'],
    )

    # 初始化对话历史
    messages = []

    conversation = 0
    max_conversation = 20 # 最大对话轮数
    while conversation < max_conversation:
        user_input = input("请输入你的问题：")

        if user_input.lower() in ['exit','quit','q','退出', '结束']:
            print('对话结束')
            break

        #收集用户输入
        messages.append({'role': 'user', 'content': user_input})
        
        #chat模式调用模型
        response = client.chat.completions.create(
            #模型名称
            model='glm-4-flash-250414',
            # 消息列表
            messages=messages,  # 聊天历史信息
            #模型参数
            temperature=0,
            # 最大token数
            max_tokens=500,      
        )
        # 收集模型输出
        ai_reply = response.choices[0].message.content
        messages.append({'role': 'assistant', 'content': ai_reply})

        # 打印结果
        print()
        print(ai_reply)
        print()
        conversation += 1
    else:
        print('对话轮数已达到最大值，结束对话。')