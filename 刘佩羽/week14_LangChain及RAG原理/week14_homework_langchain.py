import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
# 导入template中传递聊天历史信息的“占位”类
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import find_dotenv, load_dotenv
import os

# 加载文本文件转化为字符串，并赋值给一个变量
with open('homework3_prompt.txt', 'r', encoding='utf-8') as f:
    prompt = f.read()
print(prompt)

load_dotenv(find_dotenv())

# Init openai chat model

model = ChatOpenAI(
    model_name=os.getenv("MODEL_NAME"),
    base_url=os.getenv("OPENAI_API_URL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)


# Init prompt
prompt_template = ChatPromptTemplate.from_messages([
    ("system", prompt),
    MessagesPlaceholder(variable_name="messages"),
])

# Init output parser
output_parser = StrOutputParser()

# Init chain with LCEL
chain = prompt_template | model | output_parser 

# 定制存储消息的dict
# key:  sessionId会话Id（资源编号）区分不同用户或不同聊天内容
# value: InMemoryChatMessageHistory存储聊天信息list对象
store = {}

# 定义函数：根据sessionId获取聊天历史（callback 回调）
# callback 系统调用时被执行代码
def get_session_hist(session_id):
    # 以sessionid为key从store中提取关联聊天历史对象
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory() # 创建
    return store[session_id] # 检索 （langchain负责维护内部聊天历史信息）

# 在chain中注入聊天历史消息
# 调用chain之前，还需要根据sessionID提取不同的聊天历史
with_msg_hist = RunnableWithMessageHistory(
    chain, 
    get_session_history=get_session_hist,
    input_messages_key="messages"
)  # input_messages_key 指明用户消息使用key

# session_id
session_id = "abc123"

while True:
    # 用户输入
    user_input = input('用户输入的Message：')
    # 调用注入聊天历史的对象
    response = with_msg_hist.invoke(
        {
            "messages":[HumanMessage(content=user_input)]
        },
        config={'configurable':{'session_id': session_id}})

    print('AI Message:', response)
