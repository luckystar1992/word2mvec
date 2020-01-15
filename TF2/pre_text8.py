#coding:utf-8
"""
因为text8写入的时候写入的是一个句子，将其改成每1000个单词换行
"""

import sys
SEN_LENGTH = 1000

for line in sys.stdin.readlines():
    token_list = line.strip().split(" ")
    start_list = range(0, len(token_list), SEN_LENGTH)
    end_list = range(SEN_LENGTH, len(token_list) + SEN_LENGTH, SEN_LENGTH)
    for start, end in zip(start_list, end_list):
        sen = " ".join(token_list[start:end])
        sys.stdout.write(sen+"\n")
