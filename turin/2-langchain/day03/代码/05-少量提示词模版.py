import os

import langchain_openai
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv

load_dotenv()



# 创建示例
examples = [
    {"input": "2+2", "output": "4", "description": "加法运算"},
    {"input": "5-2", "output": "3", "description": "减法运算"},
]

# 创建提示模板，配置一个提示模板，将一个示例格式化为字符串
prompt_template = "你是一个数学专家,算式： {input} 值： {output} 使用： {description} "

# 这是一个提示模板，用于设置每个示例的格式  字符串提示词模版
prompt_sample = PromptTemplate.from_template(prompt_template)


prompt = FewShotPromptTemplate(
    examples=examples,   # 用到的模版
    example_prompt=prompt_sample,
    # 后缀输出    告诉大模型按照这个格式输出
    suffix="你是一个数学专家,算式: {input}  值: {output} ",
    input_variables=["input", 'output'],
)
print(prompt.format(input="2*5", output="10"))
# llm = langchain_openai.ChatOpenAI(api_key=os.getenv("api_key"),
#                                   base_url=os.getenv("base_url"),
#                                   model_name='qwen-plus')
# result = llm.invoke(prompt.format(input="2*5", output="10"))
# print(result.content)  # 使用: 乘法运算
