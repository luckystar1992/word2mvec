#coding:utf-8

"""
工具类，主要提供常用的函数使用
"""

import sys, os
import subprocess
import datetime, time
import numpy as np
import math

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    # if denom == 0:
    #     return 0
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def Pearson(vector1, vector2):
    """
    Calculating the pearson coeiffient of vec1 and vec2
    """
    avg1 = float(sum(vector1)) / len(vector1)
    avg2 = float(sum(vector2)) / len(vector2)
    sum0 = sum(map(lambda x: (x[0] - avg1) *
                   (x[1] - avg2), zip(vector1, vector2)))
    sum1 = sum(map(lambda x: (x - avg1) * (x - avg1), vector1))
    sum2 = sum(map(lambda x: (x - avg2) * (x - avg2), vector2))
    return sum0 / (math.sqrt(sum1) * math.sqrt(sum2))

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


class ArgsConfig:
    """参数的配置类，主要更新参数和保存参数"""

    def __init__(self):
        pass

    @classmethod
    def update(cls, args):
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

        # pre-trained 单一语境向量的epoch
        args.pre_epoch = 1

    @classmethod
    def save(cls, args):
        """将参数保存到当前训练结果输出目录中"""
        with open(os.path.join(args.embedding_folder, 'config.txt'), 'w') as f:
            for (args_key, args_value) in sorted(vars(args).items()):
                if isinstance(args_value, (int, float, bool, str)):
                    f.write("%20s: %10s\n" % (args_key, str(args_value)))