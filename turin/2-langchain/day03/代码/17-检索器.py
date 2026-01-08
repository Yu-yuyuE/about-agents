

import os

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
# 读取文件
loader = PyPDFLoader("财务管理文档.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)

# 将数据进行切割成块
paragraphs = text_splitter.create_documents([page.page_content for page in pages if pages])

# 创建chroma数据库，并将文本数据个向量化的数据存入
db = Chroma.from_documents(paragraphs, DashScopeEmbeddings(dashscope_api_key=os.getenv('api_key')))  # 一行代码搞定
# 实例化一个检索器
retriever = db.as_retriever()

# 我们还可以限制检索器返回的文档数量
# retriever = db.as_retriever(search_kwargs={"k": 1})

# 获取问题相关文档
docs = retriever.get_relevant_documents("会计核算基础规范")
for doc in docs:
    print(f"{doc.page_content}\n-------\n")
