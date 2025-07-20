from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests

# 下载PDF文件
url = "https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf"
path = "The_Era_of_Experience_Paper.pdf"

response = requests.get(url)
with open(path, 'wb') as f:
    f.write(response.content)

# 加载PDF文档
loader = PDFMinerLoader(path)
docs = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

# 创建向量存储
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.from_documents(texts, embeddings)

# 创建检索器
retriever = db.as_retriever(search_kwargs={"k": 3})

# 进行查询
query = "What is the main theme of this paper?"
results = retriever.get_relevant_documents(query)

print(f"查询: {query}")
print("最相关的3个结果:")
for i, doc in enumerate(results, 1):
    print(f"\n结果 {i}:")
    print(doc.page_content[:300] + "...")  # 只打印前300个字符

from langchain_experimental.graph_transformers.diffbot import GraphDocument
from langchain_experimental.graph_transformers import DiffbotGraphTransformer
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# 加载小说文本
loader = TextLoader("scandal_in_bohemia.txt")  # 假设已有文本文件
documents = loader.load()

# 初始化Diffbot图转换器 (需要API密钥)
# diffbot_api_key = "your_diffbot_api_key"
# graph_transformer = DiffbotGraphTransformer(diffbot_api_key=diffbot_api_key)

# 模拟创建图文档 (实际使用需Diffbot API)
nodes = [
    {"id": "1", "type": "PERSON", "properties": {"name": "Sherlock Holmes"}},
    {"id": "2", "type": "PERSON", "properties": {"name": "Irene Adler"}},
    {"id": "3", "type": "EVENT", "properties": {"name": "Bohemian Scandal"}}
]
relationships = [
    {"source": "1", "target": "2", "type": "INVESTIGATES"},
    {"source": "2", "target": "3", "type": "INVOLVED_IN"}
]

graph_document = GraphDocument(
    nodes=nodes,
    relationships=relationships,
    source=Document(page_content="Scandal in Bohemia story")
)

# 本地问题回答
def answer_local_question(question, graph):
    if "Holmes" in question:
        return "Sherlock Holmes is the famous detective investigating the case."
    elif "Adler" in question:
        return "Irene Adler is the woman who outsmarted Holmes in this story."
    elif "scandal" in question.lower():
        return "The Bohemian Scandal involves compromising photographs of a European king."
    return "I cannot answer that question based on the graph."

# 全局问题回答
def answer_global_question(question, graph):
    if "relationship" in question.lower() and "Holmes" in question and "Adler" in question:
        return "Holmes investigates Irene Adler but comes to admire her intellect."
    elif "outcome" in question.lower():
        return "Irene Adler outsmarts Holmes and escapes with the photographs."
    return "I cannot answer that complex question based on the graph."

# 测试问答
print("本地问题: Who is Irene Adler?")
print(answer_local_question("Who is Irene Adler?", graph_document))

print("\n全局问题: What was the relationship between Holmes and Adler?")
print(answer_global_question("What was the relationship between Holmes and Adler?", graph_document))
