import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.prompts import MessagesPlaceholder
from dotenv import find_dotenv, load_dotenv
import os

book_df = pd.read_csv('books.csv')
books_info = {row.书名: {'作者': row.作者, '简介': row.图书简介} for _, row in book_df.iterrows()}
load_dotenv(find_dotenv())
model = ChatOpenAI(
        model="gpt-3.5-turbo",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.7,
    )
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的图书助手，能够基于以下图书简介回答读者的问题：\n\n{book_summary}"),
    MessagesPlaceholder(variable_name="messages")
])
chain = prompt | model | parser
chat_histories = {}
def get_chat_history(session_id):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

with_message_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="messages"
)
if __name__ == "__main__":
    print("欢迎使用图书咨询系统！请输入您想了解的书名开始咨询。")
    while True:
        user_input = input("\n用户：").strip()
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("AI：感谢使用，再见！")
            break

        title = user_input.strip()
        if title not in books_info:
            print(f"AI：未找到书籍《{title}》，请确认书名是否正确。")
            continue

        summary = books_info[title]['简介']
        response = with_message_history.invoke(
                {
                    "book_summary": summary,
                    "messages": [HumanMessage(content="这本书讲的是什么？")]
                },
                config={"configurable": {"session_id": f"consult_{title}"}}
            )
        print(f"AI：{response}")
