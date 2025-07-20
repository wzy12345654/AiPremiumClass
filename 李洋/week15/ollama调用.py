from langchain_community.document_loaders import PDFMinerLoader
from langchain_ollama import OllamaLLM,OllamaEmbeddings
from dotenv import load_dotenv,find_dotenv
import os
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    llm = OllamaLLM(
        model='qwen2.5vl:3b',
        base_url=os.environ['url']
    )

    embeddings = OllamaEmbeddings(
        model='nomic-embed-text',
        base_url=os.environ['url']
    )

    if not os.path.exists('local_save'):
        path_url = 'https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf'
        docs = PDFMinerLoader(path_url, mode='page').load()
        loader = '\n'.join([doc.page_content for doc in docs])

        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n','\n',''],
            chunk_size=1000,
            chunk_overlap=100,
        )
        splitter_docs = splitter.split_documents(docs)


        vector_store = FAISS.from_documents(
            documents=splitter_docs,
            embedding=embeddings,
        )
        vector_store.save_local(folder_path='local_save')
        print('本地FAISS保存成功')
    else:
        vector_store = FAISS.load_local(
            folder_path='local_save',
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print('本地加载FAISS成功')

    #构建检索器
    retriever = vector_store.as_retriever()

    # 通过langchain 中 hub 里面的prompt构建项自动创建
    prompt = hub.pull('rlm/rag-prompt')

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    #构建RAG的chain
    rag_chain =(
        {"context": retriever | format_docs ,"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = rag_chain.invoke('人工智能 （AI）带来的好处')
    print(result)


