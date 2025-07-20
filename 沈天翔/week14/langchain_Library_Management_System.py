from dotenv import find_dotenv, load_dotenv
import os
from langchain_openai import ChatOpenAI # LLM调用封装
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # 对话角色：user、assistant、system
from langchain_core.output_parsers import StrOutputParser

# 导入必要包
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 导入template中传递聊天历史信息的“占位”类
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv(find_dotenv())

model = ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.7
    )

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位{Identity}，你擅长{Topic}, 请隐藏思考过程"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

parser = StrOutputParser()

chain =  prompt | model | parser

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_msg_history = RunnableWithMessageHistory(chain,
                                             get_session_history=get_session_history,
                                             input_messages_key="messages"
                                          )

def Library_Management_System(session_id, Identity, Topic):
    user_input = input("请输入你的消息：")
    print()
    response = with_msg_history.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "Identity": Identity,
            "Topic": Topic,
        },
        config={"session_id": session_id}
    )
    print(response)
    print()

if __name__ == "__main__":
    session_id = "shentx"
    Identity = "图书管理员"
    Topic = "通过图书简介为借阅读者提供咨询"

    while True:
        Library_Management_System(session_id, Identity, Topic)
