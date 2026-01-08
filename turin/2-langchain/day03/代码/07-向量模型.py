import os

from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv

load_dotenv()
# 初始化 DashScopeEmbeddings实例
embeddings = DashScopeEmbeddings(dashscope_api_key=os.getenv("api_key"), model='text-embedding-v3')


# 获取文本嵌入向量
text = '大模型'

# 嵌入文档 把文档内容转换为向量 他支持多个文档列表形式
doc_res = embeddings.embed_documents([text])
print(doc_res)

# 嵌入查询  把问题嵌入向量  一般都是一个问题
res = embeddings.embed_query(text)
print(res)