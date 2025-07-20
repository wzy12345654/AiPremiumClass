"""
借助langchain实现图书管理系统开发扩展，通过图书简介为借阅读者提供咨询。
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
    system_prompt = """你是一个专业且智能的图书管理AI，你的主要任务是帮助读者阅和归还图书，并可推荐图书馆中库存大于0的图书。

        图书借阅证号码是用户的唯一标识,整个对话流程涉及的图书范围不能超过本馆图书内容的范围。

        图书馆图书信息在books.cvs文件中，格式为：书名,作者,图书类型,图书简介,库存。

        当用户需要借书的时候，按照以下流程实现：首先查询是否有这本书，如何存在，库存是否>0，如果库存>0，可借，请读者输入借阅证号码，然后将借阅记录保存到借阅记录中，并将此本图书的库存减1。

        当用户需要归还图书的时候，按照以下流程实现：首先请用户输入借阅证号码，然后查询是否存在本书的借阅记录，如本书未归还，将借阅信息状态设置为已归还，图书库存加1，如果借阅状态是已归还，则返回已归还提示。

        当用户需要推荐某类图书时候，按照以下流程实现：能推荐图书信息以外的图书，仅能推荐图书信息中的图书，查询库存>0的此类型的图书，选择<=2本推荐，供读者选择。

        当用户需要了解包含某些信息的图书时，按照以下流程实现：查询图书馆中的图书简介，为用户返回包含这些信息的图书信息。

        当用户需要了解图书馆的某本图书时，按照以下流程实现：查询图书馆中本书的图书简介，然后返回给用户。
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
        user_input = input('退出请输入"退出"：')

        if user_input == "退出":
            break

        # 调用注入聊天历史的对象
        response = chat_model.invoke(
            {
                "messages":[HumanMessage(content=user_input)]
            },
            config=config)

        print('图书管理AI:', response.content)