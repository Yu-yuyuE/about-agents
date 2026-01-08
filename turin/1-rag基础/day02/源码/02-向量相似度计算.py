

# 余弦相似度

# 欧式距离

import numpy as np
from numpy import dot
from numpy.linalg import norm
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=os.getenv("DASHSCOPE_BASE_URL"))


def cos_sim(a, b):
    '''余弦相似度 -- 越大越相似'''
    return dot(a, b) / (norm(a) * norm(b))


def l2(a, b):
    '''欧式距离 -- 越小越相似'''
    x = np.asarray(a) - np.asarray(b)
    return norm(x)


def get_embeddings(texts, model="text-embedding-v2"):

    data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]




# 用户提的问题
query = "我国开展舱外辐射生物学暴露实验"
documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间站开展舱外辐射生物学暴露实验",
]

# 把我的文档转换成向量
dov_vecs = get_embeddings(documents)
print('文档向量大致:', dov_vecs[0][:10])
# 用户的问题转成向量
query_vec = get_embeddings([query])[0]
print('用户问题向量大致:', query_vec[:10])


# 检索用户的问题  和 知识库当中的向量相识度
# 值越大越相似   最大是1
print('余弦相识度检索  ')
print(cos_sim(query_vec, query_vec))
for doc_vec in dov_vecs:
    print(cos_sim(query_vec, doc_vec))

# 欧式距离检索  越小越相似
print('欧式距离检索  ')
print(l2(query_vec, query_vec))
for doc_vec in dov_vecs:
    print(l2(query_vec, doc_vec))

















