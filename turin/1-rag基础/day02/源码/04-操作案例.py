

import chromadb
from chromadb.config import Settings
import json
from openai import OpenAI
from dotenv import load_dotenv
import os


class MyVectorDBConnector:
    def __init__(self, collection_name):
        # 创建一个客户端
        chroma_client = chromadb.Client(Settings(allow_reset=True))
        # 创建一个 collection
        self.collection = chroma_client.get_or_create_collection(name=collection_name)


    # 把数据转换成向量
    def get_embeddings(self, texts, model="text-embedding-v2"):
        # 千问转换向量模型
        data = client.embeddings.create(input=texts, model=model).data
        return [x.embedding for x in data]


    # 添加数据到向量数据库
    def add_emb(self, instructions, outputs):
        # 把原本用户问题转换成向量
        instruction_vecs = self.get_embeddings(instructions)
        self.collection.add(
            documents=outputs,
            embeddings=instruction_vecs,
            ids=[f'id_{i}' for i in range(len(instructions))]
        )

        # {emb:instruction_vecs, document:outputs}

    def search(self, query):
        # 把用户问题转换成向量
        query_vec = self.get_embeddings([query])[0]
        result = self.collection.query(
            query_embeddings=query_vec,
            n_results=10
        )
        return result



if __name__ == '__main__':
    load_dotenv()
    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=os.getenv("DASHSCOPE_BASE_URL"))

    with open('train_zh.json', 'r', encoding='utf-8') as f:
        data = [json.loads(i) for i in f]

    # 获取前10条的问题和输出
    instructions = [entry['instruction'] for entry in data[0:10]]
    outputs = [entry['output'] for entry in data[0:10]]
    # print(data)
    vector_db = MyVectorDBConnector('demo')
    vector_db.add_emb(instructions, outputs)

    query = '得了白癜风怎么办?'
    res = vector_db.search(query)
    print(res)







