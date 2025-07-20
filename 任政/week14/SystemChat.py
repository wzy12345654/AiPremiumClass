import os
from dotenv import load_dotenv , find_dotenv
from langchain_openai import ChatOpenAI
# 对话角色
from langchain_core.messages import HumanMessage , AIMessage , SystemMessage
# 导入template中传递的聊天历史的占位符
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 导入RunnableWithMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
# 导入InMemoryChatMessageHistory ， BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory , BaseChatMessageHistory
# 导入StrOutputParser
from langchain_core.output_parsers import StrOutputParser

if __name__ == '__main__':
    # 加载.env文件中的信息
    load_dotenv(find_dotenv())

    # 初始化OpenAI Chat模型
    model = ChatOpenAI(
        model_name="glm-4-flash-250414",
        base_url=os.environ['BASE_URL'],
        api_key=os.environ['API_KEY'],
        temperature=0.7
    )
    # 设置带有占位符的模版
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个大语言模型开发工程师。尽你所能使用{lang}回答所有问题。"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    # 初始化输出解析器
    parser = StrOutputParser()
    # 构建链
    chain = prompt | model | parser
    # 定义存储消息的字典
    # key: sessionId会话Id（资源编号）区分不同用户或不同聊天内容
    # value: InMemoryChatMessageHistory存储聊天信息list对象
    store = {}
    # 定义函数：根据sessionId获取聊天历史（callback 回调）
    # callback 系统调用时被执行代码
    def get_session_hist(session_id):
        # 以sessionid为key从store中提取关联聊天历史对象
        # 如果当前的session_id 不在store里边，就新创建一个对话
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id] # 检索 （langchain负责维护内部聊天历史信息）

    # 在chain中注入聊天历史消息
    # 调用chain之前，还需要根据sessionID提取不同的聊天历史
    with_msg_hist = RunnableWithMessageHistory(
        chain,
        # 回调函数
        # 当调用chain.invoke()时，会执行get_session_history回调函数
        # 回调函数会根据session_id从store中获取对应的聊天历史对象
        # 然后将聊天历史对象作为参数传递给chain.invoke()
        # 这样，chain就可以使用聊天历史对象来构建prompt
        get_session_history=get_session_hist,
        input_messages_key="messages")  # input_messages_key 指明用户消息使用key
    print("欢迎使用大语言模型开发工程师助手")
    while True:
        # 用户输入
        # if input('是否清空历史？(y/n)') == 'y':
        #     store.clear()
        #     print('历史已清空')
        #     continue
        # user_ID = input('请输入用户ID：')
        user_input = input('用户输入的Message：')
        # 调用注入聊天历史的对象
        response = with_msg_hist.invoke(
            {
                "messages":[HumanMessage(content=user_input)],
                "lang":"中文"
            },
            config={
                "configurable":{
                    "session_id":"123"}})
        print("AI：" , response)
