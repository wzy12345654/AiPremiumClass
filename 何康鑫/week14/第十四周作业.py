from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, SystemMessage, BaseMessage
)
from langchain_core.chat_history import (
    InMemoryChatMessageHistory, BaseChatMessageHistory
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# 初始化模型
model = ChatOpenAI(
    model="glm-4-flash-250414",
    base_url=os.environ['BASE_URL'],
    api_key=os.environ['API_KEY'],
    temperature=0.7
)

# 主题定制系统提示
SYSTEM_PROMPT = """你是一个专注于{topic}领域的智能助手，需要结合历史对话为用户提供专业解答。"""

# 创建可配置的提示模板
def create_prompt(topic: str):
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.format(topic=topic)),
        MessagesPlaceholder(variable_name="messages")
    ])

# 会话历史管理
session_stores = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_stores:
        session_stores[session_id] = InMemoryChatMessageHistory()
    return session_stores[session_id]

# 创建带历史记录的链
def create_chat_chain(topic: str, session_id: str):
    prompt = create_prompt(topic)
    chain = prompt | model | StrOutputParser()
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
        history_messages_key="history"
    )

# 示例使用
if __name__ == "__main__":
    session_id = "tech_support_001"
    chat_chain = create_chat_chain("技术支持", session_id)
    
    while True:
        user_input = input("用户提问: ")
        if user_input.lower() == "exit":
            break
        
        response = chat_chain.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"session_id": session_id}}
        )
        print("AI回答:", response)

  # 图书管理系统实现 (book_system.py)
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieve import create_retrieve_chain

load_dotenv()

# 初始化模型
model = ChatOpenAI(
    model="glm-4-flash-250414",
    base_url=os.environ['BASE_URL'],
    api_key=os.environ['API_KEY'],
    temperature=0.7
)

# 加载图书数据（示例使用本地文本文件）
loader = TextLoader("books.txt", encoding="utf-8")
documents = loader.load()

# 文档分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 创建向量数据库
vectorstore = Chroma.from_documents(chunks, embedding=model.get_embeddings())

# 创建检索链
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
document_chain = create_stuff_documents_chain(model, prompt)
retrieve_chain = create_retrieve_chain(retriever, document_chain)

# 自定义问答链
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个图书管理员，需要根据提供的图书信息回答用户的问题。"),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="context")
])

qa_chain = qa_prompt | model | StrOutputParser()

# 组合检索和回答链
combined_chain = retrieve_chain | qa_chain

# 示例使用
if __name__ == "__main__":
    while True:
        query = input("用户查询: ")
        if query.lower() == "exit":
            break
        
        result = combined_chain.invoke({"messages": [HumanMessage(content=query)]})
        print("回答:", result)

  from fastapi import FastAPI, Depends
from langserve import add_routes
from chat_system import create_chat_chain
from book_system import combined_chain

app = FastAPI(title="Langchain多模态系统")

# 聊天系统路由
@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    session_id = data.get("session_id", "default")
    topic = data.get("topic", "general")
    return create_chat_chain(topic, session_id).invoke(data)

# 图书系统路由
@app.post("/book_query")
async def book_query_endpoint(request: Request):
    return combined_chain.invoke(request.json())

# 添加LangServe路由
add_routes(app, "/api")

from langchain_core.chat_history import BaseChatMessageHistory

class SessionHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history = []
        
    def add_message(self, message: BaseMessage):
        self.history.append(message)
        
    def get_messages(self) -> list[BaseMessage]:
        return self.history
    
    def clear(self):
        self.history = []
      from langchain_community.vectorstores.utils import maximal_marginal_relevance

def hybrid_retrieve(query, vectorstore, k_primary=3, k_secondary=5, fetch_k=20):
    primary_results = vectorstore.similarity_search(query, k=k_primary)
    
    if len(primary_results) < k_primary:
        secondary_results = vectorstore.similarity_search(query, k=fetch_k)
        selected = maximal_marginal_relevance(
            query_embedding=vectorstore.embeddings.embed_query(query),
            vectors=[doc.page_content for doc in secondary_results],
            k=k_secondary,
            lambda_mult=0.5
        )[:k_secondary]
        primary_results += [doc for doc in secondary_results if doc in selected]
        
    return primary_results[:k_primary]
