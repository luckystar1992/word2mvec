#coding:utf-8
"""
将下载并初步处理的wiki数据集进行清洗
"""

import sys
zh_wiki_path = '/data/zhengyuanchun/wiki/eh_extracted/AA/wiki_08'

def extract_doc_title(path):
    """将文章中的doc以及title去掉"""
    istitle = False
    for line in sys.stdin.readlines():
        if line == '\n':
            continue
        if istitle:
            # sys.stderr.write(line)
            istitle = False
            continue
        if line[0:4] == '<doc':
            istitle = True
            continue
        if line[0:4] == '</do':
            continue
        else:
            sys.stdout.write(line)
            continue


if __name__ == "__main__":
    extract_doc_title(zh_wiki_path)