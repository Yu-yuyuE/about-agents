

from langchain.prompts.chat import ChatPromptTemplate
# 导入LangChain中的ChatOpenAI模型接口
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


# template = "你是一个数学家，你可以计算任何算式"
template = "你是一个翻译专家,擅长将 {input_language} 语言翻译成 {output_language}语言."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])
# print(chat_prompt)
# 创建模型实例
model = ChatOpenAI(api_key=os.getenv("api_key"),
                   base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                   model='qwen-plus')
# 输入提示
# messages = chat_prompt.format_messages(text="我今年18岁，我的舅舅今年38岁，我的爷爷今年72岁，我和舅舅一共多少岁了？")


messages = chat_prompt.format_messages(input_language="英文", output_language="中文", text="I love Large Language Model.")
print(messages)

# 得到模型的输出
output = model.invoke(messages)
# 打印输出内容
print(output.content)

