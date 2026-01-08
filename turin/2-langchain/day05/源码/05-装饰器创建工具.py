
from langchain.tools import tool

@tool
def add_num(a, b):
    """
    这个是两个数据相加
    :param a:
    :param b:
    :return:
    """
    return a + b


print(add_num.name)
print(add_num.description)
print(add_num.args)

# add_num.run({'a': 1, 'b': 2})







# 装饰器   不改变原有的代码逻辑 可以额外添加工具

# def aa(cc):
#     def bb():
#         print('123')
#         cc()
#
#     return bb
#
# @aa
# def dd():
#     print('456')
#
#
# dd()


# b = aa(dd)  == @aa

# b()


# aa()


# @aa

#
#
# dd()




