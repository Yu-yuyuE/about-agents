from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableMap
from langchain.retrievers import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# 本地embedding模型地址
embedding_model_path = r'D:\LLM\Local_model\BAAI\bge-large-zh-v1___5'
# 初始化嵌入模型（用于文本向量化）
embeddings_model = HuggingFaceEmbeddings(
    model_name=embedding_model_path
)

# 目标 URL
url = "https://news.pku.edu.cn/mtbdnew/15ac0b3e79244efa88b03a570cbcbcaa.htm"

# 加载网页
loader = WebBaseLoader(url)
docs = loader.load()
# 创建文档分割器，并分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

persist_directory = 'db'
# 创建向量数据库
# vectorstore = Chroma.from_documents(documents=splits,
#                                     embedding=embeddings_model,
#                                     persist_directory=persist_directory)

vectorstore = Chroma(
    embedding_function=embeddings_model,
    persist_directory=persist_directory)

# 创建检索器
retriever = vectorstore.as_retriever()

relevant_docs = retriever.invoke('天才AI少女是谁')
print(relevant_docs)
print(len(relevant_docs))

# 创建llm
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

# 创建prompt模板
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""

# 由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()


print("--------------优化前-------------------")
response = chain.invoke({"question": "天才AI少女是谁"})
print(response)


print("--------------生成问题加载文档-------------------")
# 使用MultiQueryRetriever
import logging
# 打开日志
# 将日志级别设置为 INFO（显示所有日志）langchain的默认日志级别是WARNING  可以把日志等级改为INFO 就会显示生成的问题
logging.basicConfig(level=logging.INFO)
retrieval_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
)


query_docs = retrieval_from_llm.invoke("天才AI少女是谁")
print(query_docs)
print(len(query_docs))



print("--------------优化后-------------------")
chain1 = RunnableMap({
    "context": lambda x: retrieval_from_llm.invoke(input=x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

res = chain1.invoke({"question": "天才AI少女是谁"})
print(res)

