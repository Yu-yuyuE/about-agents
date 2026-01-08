
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool
import os
from aa向量存储 import faiss_conn
from dotenv import load_dotenv

load_dotenv()
# 读取数据   创建检索器
retriever = faiss_conn().as_retriever()

# 检索器工具   创建工具   给agent进行调用   大模型
retriever_tool = create_retriever_tool(
    retriever,
    "中华人民共和国民法典的一个检索器工具",
    "搜索有关中华人民共和国民法典的信息。关于中华人民共和国民法典的任何问题，您必须使用此工具!",
)

tools = [retriever_tool]


# https://smith.langchain.com/hub  提示词
prompt = hub.pull("hwchase17/openai-functions-agent")

llm = ChatOpenAI(api_key=os.getenv("api_key"),
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model='qwen-plus')
# 创建agent
agent = create_openai_functions_agent(llm, tools, prompt)
# verbose 详细模式
agent_data = AgentExecutor(agent=agent, tools=tools, verbose=True)

res = agent_data.invoke({"input": "请用中文回答，中华人民共和国民法典是什么？"})
print(res)



