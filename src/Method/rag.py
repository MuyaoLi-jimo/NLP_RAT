"""
construct a rag system
@author muyao
"""
import bs4
from langchain_community.document_loaders import WebBaseLoader,UnstructuredMarkdownLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever 
from pathlib import Path
from rich import print
import os

# 设置环境
os.environ["LANGCHAIN_TRACING_V2"] = "false"


class RAG_SYSTEM:
    def __init__(self,
                 embedding_model_path="/data/lmy/models/text2vec-base-chinese",#"/data/lmy/models/all-MiniLM-L6-v2",
                 split_method = "text_split",
                 chunk_size:int=500,chunk_overlap:int=50,add_start_index=True,
                 ):
        self.splitter = None
        if split_method == "text_split":
            self.splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=add_start_index
                )
            self.split_method = f"text_s{chunk_size}_o{chunk_overlap}"
        else:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            self.splitter=MarkdownHeaderTextSplitter(
                headers_to_split_on
            )
            self.split_method = f"markdown"
        self.embedding_model_name = str(Path(embedding_model_path).name)
        self.huggingface_embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            encode_kwargs ={'normalize_embeddings':True},
            model_kwargs={'device': 'cpu',
                          'trust_remote_code':True}
            )
        self.retriever = None 
        
    def get_retriever(self,retriever_name:str,method:str="markdown")->VectorStoreRetriever:
        vectorstore_path = Path(f"data/vectorstore") / f"{retriever_name}-{self.embedding_model_name}-{self.split_method}"
        vectorstore = None
        if vectorstore_path.exists():
            vectorstore = Chroma(persist_directory=str(vectorstore_path), 
                                 embedding_function=self.huggingface_embeddings)
        else:
            docs = self.get_docs(retriever_name,method)
            # 进行split
            docs = self.splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(documents=docs, 
                                    embedding=self.huggingface_embeddings,
                                    persist_directory=str(vectorstore_path),
                                    )
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    def retrieve(self,query:str,retriever_name:str=None):
        if retriever_name is not None:
            self.get_retriever(retriever_name)
        elif self.retriever is None:
            raise ValueError("do not define retriever")
        retrieved_docs = self.retriever.invoke(query)
        return retrieved_docs

    def get_docs(self,retriever_name:str,method="markdown"):
        # 统一先处理成markdown，
        docs = None
        loader = None
        if method == "markdown":
            markdown_fold = Path("data/crawl_docs")/retriever_name/"markdown"
            loader = DirectoryLoader(str(markdown_fold.absolute()), glob="*.md",
                                     show_progress=True,use_multithreading=True,
                                     )
            
        elif method == "crawl":
            # 测试使用
            bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
            loader = WebBaseLoader(
                web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                bs_kwargs={"parse_only": bs4_strainer},
            )
        docs = loader.load()
        #print(len(docs))
        #import pdb;pdb.set_trace()
        return docs

if __name__ =="__main__":
    rag_system = RAG_SYSTEM()
    final = rag_system.retrieve("12. (18 分) 小球 $\\mathrm{A}$ 和 $\\mathrm{B}$ 的质量分别为 $\\mathrm{m}_{\\mathrm{A}}$ 和 $\\mathrm{m}_{\\mathrm{B}}$, 且 $\\mathrm{m}_{\\mathrm{A}}>\\mathrm{m}_{\\mathrm{B}}$. 在某高度处将 $\\mathrm{A}$ 和 $\\mathrm{B}$ 先后从静止释放. 小球 $\\mathrm{A}$ 与水平地面碰撞后向上弹回, 在释放处的下方 与释放处距离为 $\\mathrm{H}$ 的地方恰好与正在下落的小球 $\\mathrm{B}$ 发生正碰. 设所有碰撞都 是弹性的, 碰撞时间极短. 求小球 $\\mathrm{A} 、 \\mathrm{~B}$ 碰撞后 $\\mathrm{B}$ 上升的最大高度.\n",retriever_name="physics")
    for d in final:
        print(d.page_content)
        print("_________________________")
    