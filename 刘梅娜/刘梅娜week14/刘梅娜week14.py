from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os

# ================== 加载环境变量 ==================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# ================== 初始化 LLM ==================
llm = ChatOpenAI(
    model="glm-4-flash-250414",
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    api_key=API_KEY,
    temperature=0.7
)

# # 创建带有记忆的对话链
# memory = ConversationBufferMemory()
# conversation = ConversationChain(
#     llm=llm,
#     memory=memory
# )

# # 多轮对话示例
# response1 = conversation.predict(input="你好，我想了解天气。")
# print(response1)  # 输出：好的，请问您想查询哪个城市的天气？

# response2 = conversation.predict(input="北京")
# print(response2)  # 输出：北京今天晴天，气温在 20°C 到 28°C 之间。

# ================== Prompt 模板 ==================
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个精通{lang}的AI助理，还能把普通话翻译为{lang}"),
    ("user", "{text}")
])

# ================== 输出解析器 ==================
output_parser = StrOutputParser()

# ================== 构建链 ==================
chain = prompt_template | llm | output_parser

# ================== 带记忆的链 ==================
with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: ChatMessageHistory(),  # 每个 session 独立的历史
    input_messages_key="text",
    history_messages_key="history"
)

config = {"configurable": {"session_id": "abc123"}}

# ================== 第一轮对话 ==================
response = with_message_history.invoke(
    {"lang": "日语", "text": "早上好！翻译过来怎么说！"},
    config=config
)
print("【日语】：" + response)

# ================== 第二轮对话 ==================
response = with_message_history.invoke(
    {"lang": "四川话", "text": "早上好！四川话怎么说！"},
    config=config
)
print("【四川话】：" + response)