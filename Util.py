#coding:utf-8

"""
工具类，主要提供常用的函数使用
"""

import sys, os
import subprocess
import datetime, time


def getFileWordCount(file_path):
    """获取某个文件的单词个数"""
    p = subprocess.Popen('wc -w ' + file_path, shell=True, stdout=subprocess.PIPE)
    out, err = p.communicate()
    line = out.splitlines()[0]

    return int(str(line, encoding='utf-8').split()[0])

def getFileBytesCount(file_path):
    """获取某个文件的bytes数目"""
    p = subprocess.Popen('wc -c ' + file_path, shell=True, stdout=subprocess.PIPE)
    out, err = p.communicate()
    line = out.splitlines()[0]

    return int(str(line, encoding='utf-8').split()[0])

def getFileName(file_path):
    """获取输入文件的名称"""
    name = os.path.splitext(file_path)[0]
    name = name.split('/')[-1]
    return name


class UpdateArgs:
    """将参数进行更新，这样在traing使用的时候，就不用管细节更新了"""

    def __init__(self):
        pass

    def update(self, args):
        """更新参数"""
        # 每次词向量的保存都保存在out_folder文件夹中
        if not os.path.exists(args.out_folder):
            os.mkdir(args.out_folder)
        timestamp = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
        embedding_folder = '{out_folder}/training_{current}'.format(
            out_folder=args.out_folder,
            current=timestamp
        )
        os.mkdir(embedding_folder)

        # 向args中添加新的参数
        args.timestamp = timestamp
        args.embedding_folder = embedding_folder


