from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

import os
from typing import List, Union
from zhipuai import ZhipuAI
from langchain_community.chat_models import ChatZhipuAI
from langchain import hub
from dotenv import find_dotenv, load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# 自定义 Zhipu Embedding 类
class ZhipuEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "embedding-3"):
        self.client = ZhipuAI(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


os.environ['PYTHONUTF8'] = '1'
# 加载文档
loader = PDFMinerLoader("week15/The Era of Experience Paper.pdf")
docs = loader.load()

# 文本拆分
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', ' '],
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# 使用智普 AI 的 Embedding 模型
os.environ["ZHIPU_API_KEY"] = ""  # 替换为你的实际密钥

embedding_model = ZhipuEmbeddings(api_key=os.environ["ZHIPU_API_KEY"])

# 构建 FAISS 向量数据库
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embedding_model)

# 保存到本地
#vectorstore.save_local('week15/local_save')


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retrieved_docs = retriever.invoke("human-centricAI")

load_dotenv(find_dotenv())
model = ChatZhipuAI(
    model="glm-4-flash-250414",
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    api_key="",
    temperature=0.7
)

#获取提示词模板
prompt = hub.pull("rlm/rag-prompt")
def format_docs(docs):
 	return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
 # retriever | format_docs 先拼接为一整段文本
 #再传给prompt形成完整的提示词
 {"context": retriever | format_docs, "question": RunnablePassthrough()}
 | prompt
 | model
 | StrOutputParser()
)


response = rag_chain.invoke("请用中文回答：解释一下人工智能的体验时代")
print(response)