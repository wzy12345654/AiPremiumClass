
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv('API_KEY')
    base_url = os.getenv('BASE_URL')
    model = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model='glm-4-plus',
        temperature=0.9
    )
    tpl = ChatPromptTemplate.from_messages([
        ('system', "你是{language}老师。"),
        MessagesPlaceholder(variable_name='messages')
    ])
    outputParser = StrOutputParser()

    def count_token(messages):
        lens = 0
        for message in messages:
            lens += len(message.content)
        return lens
    trimer = trim_messages(max_tokens=200, strategy='last',
                           token_counter=count_token)
    chain = tpl | trimer | model | outputParser
    session_store = {}

    def get_session_history(session_id):
        if session_id not in session_store:
            session_store[session_id] = InMemoryChatMessageHistory()
        return session_store[session_id]
    runnable = RunnableWithMessageHistory(
        chain, get_session_history, input_messages_key='messages')
    session_id = 123

    language = ''
    while True:
        if not language:
            language = input("请输入主题：")
            print('>>> ', language, '，输入N开启新对话，直接回车退出')

        in_msg = input("<<< ")
        if not in_msg:
            break
        if in_msg == 'N':
            language = ''
            session_id += 1
            continue
        response = runnable.invoke({
            "language": language,
            "messages": [
                HumanMessage(in_msg)
            ]
        }, config={
            'configurable': {
                'session_id': session_id
            }
        })
        print('>>> ', response)
