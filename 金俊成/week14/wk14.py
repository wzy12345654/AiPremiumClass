from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai.chat_models.base import BaseChatOpenAI

if __name__ == '__main__':
    model = BaseChatOpenAI(
        model='Qwen/Qwen2.5-7B-Instruct',
        openai_api_key='sk-jfxgwhgqleoltzwysdnkdhkflonoystyvaelmgaphmjbfvzg',
        openai_api_base='https://api.siliconflow.cn/v1',
        max_tokens=1024
    )
    # 会话历史存储
    store = {}


    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]


    # 带有占位符的prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个得力的助手。尽你所能回答所有问题。"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    parser = StrOutputParser()
    # chain 构建
    chain = prompt | model | parser

    # 调用chain之前，还需要根据sessionID提取不同的聊天历史
    with_msg_hist = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="messages")
    session_id = '123'
    while True:
        # 用户输入
        user_input = input('用户输入的Message：')
        # 调用注入聊天历史的对象
        response = with_msg_hist.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "lang": "中文"
            },
            config={'configurable': {'session_id': session_id}})

        print('AI Message:', response)
