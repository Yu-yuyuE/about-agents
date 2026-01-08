


# 句子分割       ......    。

# -*- encoding:utf-8 -*-
# import re
#
# text = "自然语言处理（NLP），作为计算机科学、人工智能与语言学的交融之地，致力于赋予计算机解析和处理人类语言的能力。在这个领域，机器学习发挥着至关重要的作用。利用多样的算法，机器得以分析、领会乃至创造我们所理解的语言。从机器翻译到情感分析，从自动摘要到实体识别，NLP的应用已遍布各个领域。随着深度学习技术的飞速进步，NLP的精确度与效能均实现了巨大飞跃。如今，部分尖端的NLP系统甚至能够处理复杂的语言理解任务，如问答系统、语音识别和对话系统等。NLP的研究推进不仅优化了人机交流，也对提升机器的自主性和智能水平起到了关键作用。"
#
# # 正则表达式匹配中文句子结束的标点符号
# sentences = re.split(r'(。|？|！|\..\..)', text)
# print(sentences)
# # 重新组合句子和结尾的标点符号
# chunks = []
# for sentence, punctuation in zip(sentences[::2], sentences[1::2]):
#     chunks.append(sentence + (punctuation if punctuation else ''))
#
# print(chunks)
# for i, chunk in enumerate(chunks):
#     print(f"块 {i + 1}: {len(chunk)}: {chunk}")



# 固定数字进行分割
# -*- encoding:utf-8 -*-

# def split_by_fixed_char_count(text, count):
#     chunks = []
#     for i in range(0, len(text), count):
#          chunks.append(text[i:i + count])
#     return chunks
#
#
# # 0 开始  280    100    0-100   100-200   200-280
#
#
# text = "自然语言处理（NLP），作为计算机科学、人工智能与语言学的交融之地，致力于赋予计算机解析和处理人类语言的能力。在这个领域，机器学习发挥着至关重要的作用。利用多样的算法，机器得以分析、领会乃至创造我们所理解的语言。从机器翻译到情感分析，从自动摘要到实体识别，NLP的应用已遍布各个领域。随着深度学习技术的飞速进步，NLP的精确度与效能均实现了巨大飞跃。如今，部分尖端的NLP系统甚至能够处理复杂的语言理解任务，如问答系统、语音识别和对话系统等。NLP的研究推进不仅优化了人机交流，也对提升机器的自主性和智能水平起到了关键作用。"
#
# # 假设我们按照每100个字符来切分文本
# chunks = split_by_fixed_char_count(text, 100)
# for i, chunk in enumerate(chunks):
#     print(f"块 {i + 1}: {len(chunk)}: {chunk}")


# 固定字符 重叠分割
# -*- encoding:utf-8 -*-
# def sliding_window_chunks(text, chunk_size, stride):
#     chunks = []
#     for i in range(0, len(text), stride):
#         chunks.append(text[i:i + chunk_size])
#     return chunks
#
#
# text = "自然语言处理（NLP），作为计算机科学、人工智能与语言学的交融之地，致力于赋予计算机解析和处理人类语言的能力。在这个领域，机器学习发挥着至关重要的作用。利用多样的算法，机器得以分析、领会乃至创造我们所理解的语言。从机器翻译到情感分析，从自动摘要到实体识别，NLP的应用已遍布各个领域。随着深度学习技术的飞速进步，NLP的精确度与效能均实现了巨大飞跃。如今，部分尖端的NLP系统甚至能够处理复杂的语言理解任务，如问答系统、语音识别和对话系统等。NLP的研究推进不仅优化了人机交流，也对提升机器的自主性和智能水平起到了关键作用。"
#
# chunks = sliding_window_chunks(text, 100, 50)  # 100个字符的块，步长为50
#
# for i, chunk in enumerate(chunks):
#     print(f"块 {i + 1}: {len(chunk)}: {chunk}")



# 递归分割
# langchain

from langchain.text_splitter import RecursiveCharacterTextSplitter
# 模块下载 pip install langchain
text = """
自 然 语 言 处 理 （N L P ），作 为 计 算 机 科 学 、 人 工 智 能 与 语 言 学 的 交 融 之 地 ， 致力于赋予计算机解析和处理人类语言的能力。在这个领域，机器学习发挥着至关重要的作用。利用多样的算法，\n\n机器得以分析、领会乃至创造我们所理解的语言。从机器翻译到情感分析，从自动摘要到实体识别，NLP的应用已遍布各个领域。随着深度学习技术的飞速进步，NLP的精确度与效能均实现了巨大飞跃。如今，部分尖端的NLP系统甚至能够处理复杂的语言理解任务，如问答系统、语音识别和对话系统等。NLP的研究推进不仅优化了人机交流，也对提升机器的自主性和智能水平起到了关键作用。
"""
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    # separators 优先级高于 chunk_size
    separators=["\n\n",'。', '']
)
chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"块 {i + 1}: {len(chunk)}: {chunk}")

# aa = [i for i in text.split("\n\n")]
# for i in aa:
#     print(i.split(' '))


# 向量
# [1 2]




