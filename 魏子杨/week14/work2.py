#查找当前目录
# import os
# print("当前工作目录:", os.getcwd())

#验证是否能找到env
# dotenv_path = "d:/ai/demo/week14/.env"
# load_dotenv(find_dotenv(dotenv_path))
# print("当前工作目录:", os.getcwd())
# print("环境变量 BASE_URL:", os.getenv("BASE_URL"))
# print("环境变量 API_KEY:", os.getenv("API_KEY"))

#模型
from dotenv import find_dotenv, load_dotenv
import os
from langchain_community.chat_models import ChatZhipuAI
#用户说的用HumanMessage，ai说的用AIMessage，不能直接使用字符串因为这要传递结构化语言
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage 

#提示词
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#转字符串
from langchain_core.output_parsers import StrOutputParser

# 记忆
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

#第一步模型创建
load_dotenv(find_dotenv())
model = ChatZhipuAI(
    model="glm-4-flash-250414",
    base_url=os.environ['BASE_URL'],
    api_key=os.environ['API_KEY'],
    temperature=0.7
)

# 第二步 提示词
 # 带有占位符的prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个智能图书管理助手，负责帮助图书馆的读者进行图书借阅、归还和个性化推荐。"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

#第三步，格式，转化为字符串返回
parser = StrOutputParser()

chain =  prompt | model | parser
session_id='aaa'
store={}
#继续包裹，要支持记忆
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
    input_messages_key="messages")  # input_messages_key 指明用户消息使用key

#添加图书读取和存储的逻辑


while True:
    session_id = input("请输入卡号：")
    user_input = input("您需要什么服务（输入 exit 退出）：")
    if user_input.lower() in ["exit", "退出"]:
        break


    response = with_msg_hist.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "lang": "中文"
        },
        config={"configurable": {"session_id": session_id}}
    )

    print('AI Message:', response)