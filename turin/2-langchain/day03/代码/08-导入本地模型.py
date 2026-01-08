

# langchain_huggingface 加载huggingface模型
from langchain_huggingface import HuggingFaceEmbeddings

# 创建嵌入模型
model_name = r'D:\LLM\Local_model\BAAI\bge-large-zh-v1___5'

# 生成的嵌入向量将被归一化, 有助于向量比较   常规
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)
text = "大模型"
query_result = embeddings.embed_query(text)
print(query_result[:5])
