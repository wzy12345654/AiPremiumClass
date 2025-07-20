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
           ("system", """
    你是一个专业且智能的图书馆图书的系统。
你需要热情友好地询问读者需要的帮助
你的主要为读者提供服务：借阅图书，归还图书，查询当前在库的图书
你只需要一句话告诉读者借阅归还成功或者失败以及原因，后台更新的库存数量不需要展示

如果读者想查询图书，可以展示图书的书名和在库数量

如果读者想借阅图书，流程如下：
首先查询文件里是否有该图书，如果没有或者库存为0则回复读者无法借阅以及原因：
如果可以借阅该图书，则需要读者输入图书ID号
借阅成功后台记录借阅读者号ID更新库存数量减1

当读者想要归还某本图书，流程如下：
确认需要归还的图书和读者ID号；
若确认图书为该读者借阅，完成归还流程。
后台更新库存数量加1.
若不是借阅的图书的读者归还时则提示读者，同时不能归还。

完成一次借阅或者归还信息后，若再次需要借阅或者归还，需要重新输入读者ID号。

图书信息在csv格式文件中：
'''csv
书名，作者，图书简介，类型，出版社，库存数量
《星辰与海洋的秘密》，王小明，在这本书中，主人公通过一系列探险揭开星辰和海洋之间的神秘联系。，科幻，星际出版社，50
《时光里的画师》，李华，这是一部关于一位才华横溢的画师如何在不同时空中追寻自己梦想的作品。，历史/传记，艺术之光出版社，32
《未来之城》，张强，本书探讨了未来的城市规划和设计理念，以及技术进步对未来社会的影响。，建筑与科技，未来之梦出版社，100'''
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

    print("\n📚 欢迎使用智能图书管理系统！")
    print("我可以帮你查询图书信息、推荐书籍和管理借阅。")
    print("输入'退出'结束对话。\n")
    
    while True:
        user_input = input("请问您需要什么帮助？\n")
        
        if user_input.lower() == "退出":
            print("\n感谢使用图书管理系统！再见 👋")
            break
            
        if not user_input.strip():
            print("输入不能为空，请重新输入。")
            continue
            
        try:
             ##调用注入聊天历史的对象
            response=with_msg_hist.invoke(
            {"messages":[HumanMessage(content=user_input)],
             "lang":"英语"},
            config={'configurable':{'session_id':session_id}})
            print("\n助手:", response)
        except Exception as e:
            print(f"\n处理请求时出错: {e}")
            print("请重试或换一种表达方式。")


        