
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
load_dotenv()

llm = ChatOpenAI(api_key=os.getenv("api_key"),
                 base_url=os.getenv("base_url"),
                 model_name="qwen-turbo")

# 直接提供问题，并调用llm
response = llm.invoke("你好我是柏汌")
# print(response)
# print("=" * 50)
print(response.content)

response = llm.invoke("我是谁?")

print(response.content)


