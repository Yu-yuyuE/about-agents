
from langchain.tools import tool


@tool
def add_num(a, b):
    '''
    用于计算两个数的和
    '''
    return a + b



print(add_num.run({"a": 1, "b": 2}))
print(add_num.name)
print(add_num.description)
print(add_num.args)


