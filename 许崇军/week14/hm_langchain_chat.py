from dotenv import find_dotenv, load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    # Initialize the OpenAI Chat model
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.8,
        max_tokens=100,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个很擅长有来有回式聊天的助手。尽你所能围绕这个{topic}主题进行对话。要能反问、提问、回答问题，不要让用户觉得你在聊天。"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    parser = StrOutputParser()
    chain = prompt | model | parser
    # 存储结构扩展为 dict of dict，支持每个 session 独立的 history 和 topic
    store = {
        # 示例格式：
        # "abc123": {"history": InMemoryChatMessageHistory(), "topic": "焦虑"},
    }
    def get_session_hist(session_id):
        if session_id not in store:
            default_topic = input(f"会话 {session_id} 首次启动，请输入你想聊的主题（如“死亡”、“焦虑”等）: ")
            store[session_id] = {
                "history": InMemoryChatMessageHistory(),
                "topic": default_topic
            }
        return store[session_id]["history"], store[session_id]["topic"]
    with_msg_hist = RunnableWithMessageHistory(
        chain,
        get_session_history=lambda sid: get_session_hist(sid)[0],
        input_messages_key="messages"
    )
    session_id = "abc123"
    print("欢迎！请输入内容开始聊天，或输入 'exit' / 'quit' 退出程序。")
    while True:
        user_input = input('\033[94m用户:\033[0m ')
        if user_input.lower() in ["exit", "quit"]:
            print("\033[92mAI: 再见！\033[0m")
            break
        # 获取当前 session 的 topic
        _, current_topic = get_session_hist(session_id)

        response = with_msg_hist.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "topic": current_topic
            },
            config={'configurable': {'session_id': session_id}}
        )
        print(f'\033[92mAI: \033[0m{response}\n')
