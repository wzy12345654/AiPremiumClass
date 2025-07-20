from dotenv import find_dotenv, load_dotenv
import os

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

if __name__ == '__main__':
    
    #加载环境变量
    load_dotenv(find_dotenv())

    # template对象
    prompt_temp = ChatPromptTemplate.from_messages(
        [
            ("system", "{lang}"),
            MessagesPlaceholder(variable_name="messages")  # 占位符，用于注入聊天历史
        ]
    )

    #加载模型
    model = ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ["base_url"],
        api_key=os.environ["api_key"],
        temperature=0.7
    )

    # LLM结果解析器
    paser = StrOutputParser()

    # langchain表达式语言 (LCEL) 构建 chain
    chain = prompt_temp | model | paser

    #定制存储消息的dict
    # key:  sessionId会话Id（资源编号）区分不同用户或不同聊天内容
    # value: InMemoryChatMessageHistory存储聊天信息list对象
    store = {}

    #定义函数
    # 根据sessionId获取聊天历史（callback 回调）
    def get_session_hist(session_id):
        # 以sessionid为key从store中提取关联聊天历史对象
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()

        return store[session_id]  # 检索 （langchain负责维护内部聊天历史信息）
    
    # 在chain中注入聊天历史消息
    # 调用chain之前，还需要根据sessionID提取不同的聊天历史
    #管理对话历史并自动将其注入到链(Chain)的执行中
    with_meg_his=RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_hist,
        input_messages_key="messages"  # input_messages_key 指明用户消息使用key
    )

    # session_id
    session_id = "abc123"
    
    while True:
        user_input = input("请输入：")
        response=with_meg_his.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "lang": "你是一个图书管理系统，请根据用户输入的内容生成图书信息",
            },
            config={"configurable":{"session_id":session_id}}
        )

        print("AImessage:", response)

        store[session_id].add_message(HumanMessage(content=user_input))
        store[session_id].add_message(AIMessage(content=response))

    

