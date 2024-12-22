"""
construct a rag system
@author muyao
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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
    rag= RAG_SYSTEM()
    