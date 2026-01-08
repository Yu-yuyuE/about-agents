
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=os.getenv("DASHSCOPE_BASE_URL"))


def get_embeddings(texts, model="text-embedding-v3"):
    #  texts 是一个包含要获取嵌入表示的文本的列表，
    #  model 则是用来指定要使用的模型的名称
    #  生成文本的嵌入表示。结果存储在data中。
    data = client.embeddings.create(input=texts, model=model).data
    print(data)
    # 返回了一个包含所有嵌入表示的列表
    return [x.embedding for x in data]

vec = get_embeddings(["你好"])

print('第一个向量:', vec[0])
print('向量的维度:', len(vec[0]))
print(vec[0][:10])






