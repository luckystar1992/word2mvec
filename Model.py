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
        embedding_folder = self.args.embedding_folder
        print(" Saving {out_folder}".format(out_folder=embedding_folder))

        # 先将此次的训练参数保存下来
        with open(os.path.join(embedding_folder, 'config.txt'), 'w') as f:
            for (args_key, args_value) in sorted(vars(self.args).items()):
                if isinstance(args_value, (int, float, bool, str)):
                    f.write("%20s: %10s\n" % (args_key, str(args_value)))

        #  开始保存词向量
        if self.args.binary:
            embedding_path = os.path.join(embedding_folder, 'wv_epoch{epoch}.bin'.format(epoch=epoch))
            with open(embedding_path, 'wb') as f_out:
                f_out.write(('%d %d\n' % (len(self.net1), self.args.embedding_size)).encode())
                for token, vector in zip(self.vocab, self.net1):
                    f_out.write(('%s %s\n' % (token.word, ' '.join([str(s) for s in vector]))).encode())
        else:
            embedding_path = os.path.join(embedding_folder, 'wv_epoch{epoch}.txt'.format(epoch=epoch))
            with open(embedding_path, 'w') as f_out:
                f_out.write('%d %d\n' % (len(self.net1), self.args.embedding_size))
                for token, vector in zip(self.vocab, self.net1):
                    f_out.write('%s %s\n' % (token.word, ' '.join([str(s) for s in vector])))

    def saveEmbeddingOut(self, epoch):
        """保存第二个词向量"""

    def saveSenses(self):
        """保存上下文向量"""
        raise NotImplementedError


class MultiSenseModel(SingleModel):
    """多语境词向量模型"""
    
    def __init__(self, args, vocab):
        super(MultiSenseModel, self).__init__(args, vocab)
        self.senses_number = args.senses
        print(args.senses)

    def init_model(self):
        """初始化模型参数，包括模型权重，多语境词向量，多语境向量"""
        # [V] 记录当前token的不同senses共有多少个count
        tmp = np.zeros(self.vocab_size)
        senses_count = np.ctypeslib.as_ctypes(tmp)
        self.senses_count = Array(senses_count._type_, senses_count, lock=False)

        # [V, senses + 1, dim] 模型的多语境词向量
        # 多语境的第一个词向量为main-embedding
        tmp = np.random.uniform(low=-0.5/self.embedding_size,
                                high=0.5/self.embedding_size,
                                size=(self.vocab_size, self.senses_number + 1, self.embedding_size))
        embedding = np.ctypeslib.as_ctypes(tmp)
        self.embedding = Array(embedding._type_, embedding, lock=False)

        # [V, senses +1 , dim] 模型的多语境词向量对应的语境信息
        # 语境的第一个词向量为main-sense
        tmp = np.random.uniform(low=-0.5/self.embedding_size,
                                high=0.5/self.embedding_size,
                                size=(self.vocab_size, self.senses_number + 1, self.embedding_size))
        senses = np.ctypeslib.as_ctypes(tmp)
        self.senses = Array(senses._type_, senses, lock=False)

        # [V, dim] 矩阵作为第二层权重
        tmp = np.zeros(shape=(self.vocab_size, self.embedding_size))
        weights = np.ctypeslib.as_ctypes(tmp)
        self.weights = Array(weights._type_, weights, lock=False)

    def updateEembedding(self, token_id, sense_index, gradient):
        """更新对应语境下的词向量 + 更新对应语境向量"""
        self.embedding[token_id][sense_index] += gradient
        self.senses[token_id][sense_index] += gradient

    def updateWeights(self, token_id, gradient):
        self.weights[token_id] += gradient

    def saveEmbedding(self, epoch):
        """保存词向量，并将对应的语境向量也保存， 分开文件保存"""
        embedding_folder = self.args.embedding_folder
        print(" Saving {out_folder}".format(out_folder=embedding_folder))

        if not self.args.binary:
            embedding_path = os.path.join(embedding_folder, 'wv_epoch{epoch}.bin'.format(epoch=epoch))
            sense_path = os.path.join(embedding_folder, 'sv_epoch{epoch}.bin'.format(epoch=epoch))
            with open(embedding_path, 'w') as f_wv, open(sense_path, 'w') as f_sv:
                f_wv.write("%d %d\n" % (len(self.embedding), self.args.embedding_size))
                for token_index, (token, sense_count) in enumerate(zip(self.vocab, self.senses_count)):
                    for sense_index, count in enumerate(sense_count):
                        if count == 0:
                            break
                    valid_embeddings = self.embedding[token_index][:sense_index]
                    valid_embeddings = valid_embeddings.flatten().tolist()
                    f_wv.write("%s %d %s %s\n" % (
                        token.word,
                        sense_index,
                        " ".join([str(item) for item in self.senses_count[token_index][:sense_index]]),
                        " ".join([str(item) for item in valid_embeddings])
                    ))

                    valid_sense = self.senses[token_index][:sense_index]
                    valid_sense = valid_sense.flatten().tolist()
                    f_sv.write("%s %d %s %s\n" % (
                        token.word,
                        sense_index,
                        " ".join([str(item) for item in self.senses_count[token_index][:sense_index]]),
                        " ".join([str(item) for item in valid_sense])
                    ))
