"""
construct a rag system
@author muyao
"""
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from pathlib import Path
import os

# 设置环境
os.environ["LANGCHAIN_TRACING_V2"] = "false"



class RAG_SYSTEM:
    def __init__(self,
                 embedding_model_path="/data/lmy/models/all-MiniLM-L6-v2",
                 chunk_size:int=1000,chunk_overlap:int=200,add_start_index=True,
                 ):
        self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=add_start_index
            )
        self.huggingface_embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            encode_kwargs ={'normalize_embeddings':True},
            model_kwargs={'device': 'cpu',
                          'trust_remote_code':True}
            )
        self.retriever = None 
        
    def get_retriever(self,retriever_name:str)->VectorStoreRetriever:
        vectorstore_path = Path(f"data/vectorstore") / retriever_name
        vectorstore = None
        if vectorstore_path.exists():
            vectorstore = Chroma(persist_directory=str(vectorstore_path), 
                                 embedding_function=self.huggingface_embeddings)
        else:
            docs = self.get_docs(retriever_name)
            # 进行split
            all_splits = self.text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(documents=all_splits, 
                                    embedding=self.huggingface_embeddings,
                                    persist_directory=str(vectorstore_path),
                                    )
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    
    def retrieve(self,query:str,retriever_name:str=None):
        if retriever_name is not None:
            self.get_retriever(retriever_name)
        elif self.retriever is None:
            raise ValueError("do not define retriever")
        retrieved_docs = self.retriever.invoke(query)
        return retrieved_docs

    def get_docs(self,retriever_name:str):
        # Only keep post title, headers, and content from the full HTML.
        bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs={"parse_only": bs4_strainer},
        )
        docs = loader.load()
        return docs

if __name__ =="__main__":
    rag_system = RAG_SYSTEM()
    final = rag_system.retrieve("What is Task Decomposition?",retriever_name="2023-06-23-agent")
    print(final[0].page_content)
    