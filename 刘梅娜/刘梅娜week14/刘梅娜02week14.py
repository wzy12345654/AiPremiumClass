from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# 自定义 Embedding 类
from zhipuai import ZhipuAI
from langchain_core.embeddings import Embeddings
from typing import List

class ZhipuEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "embedding-2"):
        self.client = ZhipuAI(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            results.append(response.data[0].embedding)
        return results

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# 图书数据示例
books = [
    {"title": "机器学习基础", "description": "本书介绍了机器学习的基本概念和算法..."},
    {"title": "深度学习实战", "description": "本书讲解了深度学习的实际应用案例..."}
]

# 提取描述文本
descriptions = [book["description"] for book in books]

# 使用自定义 Embedding
embeddings = ZhipuEmbeddings(api_key=API_KEY)

# 构建向量数据库
vectorstore = FAISS.from_texts(descriptions, embeddings)

# 初始化 LLM
llm = ChatOpenAI(
    model="glm-4-flash-250414",
    api_key=API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 创建 Prompt（用于问答）
prompt = ChatPromptTemplate.from_messages([
    ("system", "请根据以下上下文回答问题：\n\n{context}"),
    ("user", "{input}")
])

# 创建文档合并链
combine_docs_chain = create_stuff_documents_chain(llm, prompt)

# 创建检索链
retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

# 用户提问测试
query = "有没有关于深度学习的书籍？"
response = retrieval_chain.invoke({"input": query})
print(response["answer"])