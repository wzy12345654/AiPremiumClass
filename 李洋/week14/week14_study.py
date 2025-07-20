from datasets.fingerprint import fingerprint_rng
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv,find_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
#导入聊天历史应用的包
from langchain_core.chat_history import BaseChatMessageHistory,InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    chat = ChatOpenAI(
        model_name='glm-4-flash-250414',
        api_key=os.environ['api_key'],
        base_url=os.environ['base_url'],
        temperature=0.7
    )
    prompts = ChatPromptTemplate.from_messages(
        [
            ('system','你是一个图书管理助手，通过图书简介为借阅读者使用{lang}提供咨询'),
            MessagesPlaceholder(variable_name='messages')
        ],

    )
    parser = StrOutputParser()

    chain = prompts | chat | parser

    # message = [
    #     SystemMessage(content='你好！我叫 神奇的一天'),
    #     HumanMessage(content='你是一个重名的AI聊天助手'),
    #     AIMessage(content='你好！很高兴认识你，神奇的一天。虽然我是一个AI聊天助手，但很高兴能和你进行交流。请问有什么我可以帮助你的吗？'),
    #     HumanMessage(content='你还记得我叫什么嘛')
    #
    # ]
    #
    # response = chat.invoke(message)
    #
    # print(response.content)

    #定制一个消息的字典
    #key/value: sessionId 会话 Id(资源编号) 区分不用用户或不用聊天内容 每个用户都有自己的ID
    store = {}

    #定义函数名称
    def get_session_history(session_id):
        #以sessionID 为Key从store中提取关联聊天历史对象
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    #在我们的chain中 聊天记录存储
    with_mas_hist = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key='messages',
    )

    session_id = 'abc123'

    while True:
        user_input = input('书籍描述:')
        response = with_mas_hist.invoke(
            {
                "messages":[HumanMessage(content=user_input)],
                "lang":"中文"
            } ,
            config={'configurable': {'session_id': session_id}})

        print('AI 图书管理员:',response)