"""
通过langchain实现特定主题聊天系统，支持多轮对话。
"""
import os
from dotenv import load_dotenv,find_dotenv
from langchain_openai import ChatOpenAI # LLM调用封装
from langchain_core.messages import HumanMessage,SystemMessage,trim_messages
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory,InMemoryChatMessageHistory  #记录历史对话
from langchain_core.runnables import RunnableWithMessageHistory,RunnablePassthrough  #RunnableWithMessageHistory历史消息调用 ; RunnablePassthrough 为了确保消息传递到后面
from operator import itemgetter
import dashscope

store={}
def get_session_history(session_id:str) -> BaseChatMessageHistory:
    """
    根据给定的会话 ID 获取对应的聊天历史记录对象。
    如果该会话 ID 不存在对应的聊天历史记录，则创建一个新的内存存储的聊天历史记录对象。
    参数:
    session_id (str): 用于唯一标识一个会话的字符串。
    返回:
    BaseChatMessageHistory: 与给定会话 ID 对应的聊天历史记录对象。
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def custom_token_counter(messages):
    """
    自定义令牌计数函数，使用 dashscope 计算消息的令牌数量。
    """
    dashscope_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, HumanMessage):
            role = "user"
        else:
            role = "assistant"
        dashscope_messages.append({"role": role, "content": msg.content})

    # 使用 count_tokens_only 参数计算token数
    response = dashscope.Generation.call(
        model="qwen-max",
        messages=dashscope_messages,
        count_tokens_only=True  # 仅计算token，不生成回复
    )
    if response.status_code == 200:
        return response.usage.total_tokens
    else:
        raise Exception(f"Token counting failed: {response.message}")

if __name__ == '__main__':
    # 0. 加载环境变量
    load_dotenv(find_dotenv())
    dashscope.api_key = os.environ['api_key']

    # 1. 加载大预言模型
    llm = ChatOpenAI(
        model_name='qwen-max',
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.1,
        request_timeout=600
    )

    # 2. 定义提示模板
    system_prompt = """你是一个智能文具电商客服，用户输入的Message都是用户在电商平台的购物相关问题，你要根据用户的Message回复用户。
        出售的商品仅限于文具用品，不包含服饰，书籍。如果用户咨询的商品不在出售范围内，你要礼貌的告诉他，谢谢。
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])

    # 3. 管理对话历史记录
    trimmer = trim_messages(
        max_tokens=1024,
        strategy="last", #策略取最后
        token_counter = custom_token_counter,
        include_system = True,
        allow_partial = False, #不允许不完整
        start_on = "human" #从人类开始
    )
    # 4. 定义chain
    chain = (
        RunnablePassthrough.assign(messages=itemgetter("messages")| trimmer)
        | prompt
        | llm
    )

    # 5. 带历史记录的对话模型
    chat_model = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages")
    config = {"configurable": {"session_id": "123"}}  #配置信息
    
    # 6. 对话
    while True:
        # 用户输入
        user_input = input('输入您的问题，如需退出请输入"退出"：')

        if user_input == "退出":
            break

        # 调用注入聊天历史的对象
        response = chat_model.invoke(
            {
                "messages":[HumanMessage(content=user_input)]
            },
            config=config)

        print('电商智能客服:', response.content)
