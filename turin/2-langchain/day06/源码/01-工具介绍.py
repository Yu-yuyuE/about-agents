


# langchain发布了新版本    1.0 移除

from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv


load_dotenv()


tool = TavilySearchResults(top_k=1)

# 工具名称
print(tool.name)

# 工具的描述
print(tool.description)

# 输入内容格式   # 默认json
print(tool.args)


# 工具使用
aa = tool.run("你好")
print(aa)







