import os
from dotenv import load_dotenv,find_dotenv
from langchain_openai import ChatOpenAI ##llm调用封装
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage ##对话角色 user ，assisant，system
from langchain_core.chat_history import BaseChatMessageHistory,InMemoryChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

##导入template中传递聊天历史信息的占位类
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder


if __name__ =='__main__':
    load_dotenv(find_dotenv())
   


    ##创建调用客户端
    model=ChatOpenAI(
        model='glm-4-flash-250414',
        api_key=os.environ['api_key'],
        base_url=os.environ['base_url'],
        temperature=0.7
    )

    ##带有占位符的prompt
    prompt=ChatPromptTemplate.from_messages(
        [
           ("system","""
你是一位专业的旅行规划助手。
- 首先询问用户出发地、目的地、旅行天数、预算和兴趣偏好
- 生成详细的每日行程安排，包括交通方式对应费用耗时、景点门票，景点特色，景点游览时间、餐饮推荐，每天每个景点的推荐指数排序
- 提供符合用户预算的住宿选择建议
- 根据用户反馈调整方案
- 提供当地文化习俗、天气和安全注意事项
- 最后生成一个可总结的行程概览

请确保回答友好、信息丰富，并提供具体的建议和实用信息。
如果用户询问的信息与旅行无关，请礼貌地提醒用户你专注于旅行规划。
""",),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    parser=StrOutputParser()

    ##chain构建
    chain=prompt|model|parser

    ###定制存储消息的dict
    ##key：sessionID会话Id（资源编号）区分不同用户或不同聊天内容
    ##value:InMemoryCHatMesssageHistory存储聊天信息

    store={}

    ##定义函数：根据sessionId获取聊天历史(callback回调)
    ##callback系统调用时被执行的代码
    def get_session_hist(session_id):
        #以sessionId为key从store中提取关联聊天历史对象
        if session_id not in store:
            store[session_id]=InMemoryChatMessageHistory()
        return store[session_id]

    
    ##在chain中注入聊天历史消息
    ## 调用chain之前，还需要根据sessionID提取不同的聊天历史
    with_msg_hist= RunnableWithMessageHistory(chain,
                                              get_session_history=get_session_hist,
                                              input_messages_key="messages")

    ##session_id
    session_id="abc123"

    # 欢迎信息
    print("\n🌍 欢迎使用旅行规划助手！我可以帮你规划完美的旅行行程。")
    print("请告诉我你想去哪里旅行，以及你的具体需求（如天数、预算等）。")
    print("输入'退出'结束对话。\n")
 
    while True:
        #用户输入
        user_input=input('用户输入的message:')
        if user_input=='退出':
            break
        ##调用注入聊天历史的对象
        response=with_msg_hist.invoke(
            {"messages":[HumanMessage(content=user_input)],
             "lang":"英语"},
            config={'configurable':{'session_id':session_id}})

        print("用户输入:",user_input)
        print('AI Message:',response)

        