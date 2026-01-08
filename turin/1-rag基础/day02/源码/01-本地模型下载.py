# 模型下载
# from modelscope import snapshot_download
# # pip install modelscope   # 2-3g
# # 第一个是需要下载的模型名称
# # 第二个是下载到本地的路径
# model_dir = snapshot_download("BAAI/bge-large-zh-v1.5", cache_dir="D:\LLM\Local_model")


# 本地向量模型调用
# 加载本地模型
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(r'D:\LLM\Local_model\BAAI\bge-large-zh-v1___5')


emb = model.encode(["你好"])
print(emb.shape)  # 查看转换的向量个数    向量维度是1024











