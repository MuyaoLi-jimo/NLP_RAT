"""
construct a rag system
@author muyao
"""
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import getpass
import os

# 设置环境
os.environ["LANGCHAIN_TRACING_V2"] = "false"
"""
# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# spliting the text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=all_splits, 
                                    embedding=HuggingFaceEmbeddings(model_name="/data/lmy/models/all-MiniLM-L6-v2")
                                    ,persist_directory="1")
"""
vectorstore = Chroma(persist_directory='1', embedding_function='')
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")

print(len(retrieved_docs))

print(retrieved_docs[0].page_content)

class RAG_SYSTEM:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        self.huggingface_embeddings = HuggingFaceEmbeddings(
            model_name="/Users/jimoli/Desktop/my_hw/NLP RAG/models",
            encode_kwargs ={'normalize_embeddings':True},
            model_kwargs={'device': 'cpu',
                          'trust_remote_code':True}
        
        )
    def create_db():
        pass

if __name__ =="__main__":
    pass
    