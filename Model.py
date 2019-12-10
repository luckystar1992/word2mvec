#coding:utf-8

"""
模型类
"""

import sys, os
import struct
import time
import numpy as np
from multiprocessing import Array, Value

class SingleModel:

    def __init__(self, args, vocab):
        """
        :param args:  参数列表
        :param vocab: 词汇表对象
        """
        self.args = args
        self.vocab = vocab
        self.embedding_size = args.embedding_size
        self.vocab_size = len(self.vocab)

    def init_model(self):
        """初始化模型的参数"""
        # [V, dim] 矩阵作为第一层权重
        # 这里的net1和net2使用的是内存共享的机制，在train中init_process的时候也进行了一次转化
        # 但是这里的net1和net2指向的是同一块内存
        tmp = np.random.uniform(low=-0.5/self.embedding_size,
                                high=0.5/self.embedding_size,
                                size=(self.vocab_size, self.embedding_size))
        net1 = np.ctypeslib.as_ctypes(tmp)
        self.net1 = Array(net1._type_, net1, lock=False)

        # [V, dim] 矩阵作为第二层权重
        tmp = np.zeros(shape=(self.vocab_size, self.embedding_size))
        net2 = np.ctypeslib.as_ctypes(tmp)
        self.net2 = Array(net2._type_, net2, lock=False)

    def getNetMean(self, index):
        return np.mean(self.net1[index])


    def updateNet1(self, index, gradient):
        """更新网络层1的权重"""
        self.net1[index] += gradient

    def updateNet2(self, index, gradient):
        """更新网络层2的权重"""
        self.net2[index] += gradient

    def saveEmbedding(self, epoch):
        """保存词向量"""
        out_folder = self.args.out_folder
        print(" Saving {out_folder}".format(out_folder=out_folder))

        # 先将此次的训练参数保存下来
        with open(os.path.join(out_folder, 'config.txt'), 'w') as f:
            for (args_key, args_value) in sorted(vars(self.args).items()):
                if isinstance(args_value, (int, float, bool, str)):
                    f.write("%20s: %10s\n" % (args_key, str(args_value)))

        #  开始保存词向量
        if self.args.binary:
            embedding_path = os.path.join(out_folder, 'wv_epoch{epoch}.bin'.format(epoch=epoch))
            with open(embedding_path, 'wb') as f_out:
                f_out.write(('%d %d\n' % (len(self.net1), self.args.embedding_size)).encode())
                for token, vector in zip(self.vocab, self.net1):
                    f_out.write(('%s %s\n' % (token.word, ' '.join([str(s) for s in vector]))).encode())
        else:
            embedding_path = os.path.join(out_folder, 'wv_epoch{epoch}.txt'.format(epoch=epoch))
            with open(embedding_path, 'w') as f_out:
                f_out.write('%d %d\n' % (len(self.net1), self.args.embedding_size))
                for token, vector in zip(self.vocab, self.net1):
                    f_out.write('%s %s\n' % (token.word, ' '.join([str(s) for s in vector])))


    def saveContentEmbedding(self):
        """保存上下文向量"""
        raise NotImplementedError



