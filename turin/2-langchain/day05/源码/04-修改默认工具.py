

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field

# 初始化工具 可以根据需要进行配置
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)


class WikiInputs(BaseModel):
    """维基百科工具的输入。"""

    query: str = Field(
        description="维基百科中的查询，字数应在3个字以内"
    )


tool = WikipediaQueryRun(
    name="wiki-tool",
    description="在维基百科中查找内容",
    args_schema=WikiInputs,
    api_wrapper=api_wrapper,
    return_direct=True,
)

# 工具默认名称
print("name:", tool.name)
# 工具默认的描述
print("description:", tool.description)
print(tool.run("langchain"))
