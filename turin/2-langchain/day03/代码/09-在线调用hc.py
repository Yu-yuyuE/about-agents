

import os

from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

ENDPOINT_URL = "HuggingFaceH4/zephyr-7b-beta"
# ENDPOINT_URL = "deepseek-ai/DeepSeek-R1"
HF_TOKEN = os.getenv('HF_TOKEN')

llm = HuggingFaceEndpoint(
    endpoint_url=ENDPOINT_URL,
    # max_new_tokens=30,  限制生成的最大 token 数量为 30 个
    typical_p=0.95,     # 控制输出文本的多样性，避免生成太过常见或太过罕见的 tokens
    temperature=0.01,
    repetition_penalty=1.03,    # 对重复出现的 tokens 施加惩罚，避免生成重复的内容
    huggingfacehub_api_token=HF_TOKEN
)

print(llm.invoke("解释langchain是什么？"))
# 生成token时需要把权限都点上
