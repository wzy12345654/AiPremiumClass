from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
# 创建向量数据库（内存中）对chunks进行编码
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

if __name__ == '__main__':
    # 加载环境变量
    load_dotenv(find_dotenv())

    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0613",
        api_key=os.environ["API_KEY"],
        base_url=os.environ["API_URL"],
    )

    # OpenAIEmbeddings作用是使用OpenAI的embedding模型对文本进行编码
    embedding = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=os.environ["API_KEY"],
        base_url=os.environ["API_URL"],
    )

    if os.path.exists("faiss_store"):
        # 加载向量数据库
        vectorstore = FAISS.load_local(
            "faiss_store",
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )
        print('加载faiss数据库本地化记录成功！')
    else:
        # WebBaseLoader作用是从网页中提取文本，
        # 并返回一个Document对象，供langchain使用。
        # loader = WebBaseLoader(
        #     web_path='https://hawstein.com/2014/03/06/make-thiner-make-friend-with-time/'
        # )
        loader = PDFMinerLoader(
            file_path="https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf",
        )

        # 使用WebBaseLoader加载网页内容
        docs = loader.load()

        # documents text split
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " "],
            chunk_size=1000,  # 每个chunk的最大字符数
            chunk_overlap=100  # 每个chunk之间的重叠字符数
        )

        splite_doc = text_splitter.split_documents(docs)
        # print(splite_doc[0], len(splite_doc))
        
    
        # FAISS作用是使用FAISS库创建一个向量数据库
        vectorstore = FAISS.from_documents(
            documents=splite_doc,
            embedding=embedding
        )
        # 将向量数据库保存到本地
        vectorstore.save_local("faiss_store")
        print('faiss数据库本地化保存成功！')
    # retrieve: 使用向量数据库进行检索
    # 1. 构建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # 检索5个最相关的文档
    # 2. 构建rag chain
    # 2.1 构建prompt
    prompt = hub.pull("rlm/rag-prompt")
    print(f'prompt: {prompt}')
    # 2.2 构建rag chain

    def format_docs(docs):
        """Format the docs for the prompt."""
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    # rag检索
    response = rag_chain.invoke("Welcome to the Era of Experience主要讲了什么")
    print(response)
    print("运行结束！")
