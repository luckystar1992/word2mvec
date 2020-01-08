#coding:utf-8

"""
模型类
"""

import sys, os
import struct
import time
import tqdm
import numpy as np
from multiprocessing import Array, Value
import Util

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

    def init_model(self):
        """
            初始化模型参数，包括模型权重，多语境词向量，多语境向量
            向量包括一个Meta-Embedding和众多Senses-Embedding
        """
        # [V] 记录每个token共有多少不同的count
        tmp = np.zeros(self.vocab_size, dtype='int32')
        senses_count = np.ctypeslib.as_ctypes(tmp)
        self.senses_count = Array(senses_count._type_, senses_count, lock=False)

        # [V] 记录每个token的每个sense被访问过多少次
        tmp = np.zeros(shape=(self.vocab_size, self.senses_number + 1), dtype='int32')
        senses_access = np.ctypeslib.as_ctypes(tmp)
        self.senses_access = Array(senses_access._type_, senses_access, lock=False)

        # [V, dim] 模型的主向量 + 模型的主向量对应的
        tmp = np.random.uniform(low=-0.5/self.embedding_size,
                                high=0.5/self.embedding_size,
                                size=(self.vocab_size, self.embedding_size))
        main_embedding = np.ctypeslib.as_ctypes(tmp)
        self.main_embedding = Array(main_embedding._type_, main_embedding, lock=False)

        tmp = np.zeros(shape=(self.vocab_size, self.embedding_size))
        main_sense = np.ctypeslib.as_ctypes(tmp)
        self.main_sense = Array(main_sense._type_, main_sense, lock=False)

        # [V, senses, dim] 模型的多语境词向量
        # 多语境的第一个词向量为main-embedding
        tmp = np.random.uniform(low=-0.5/self.embedding_size,
                                high=0.5/self.embedding_size,
                                size=(self.vocab_size, self.senses_number, self.embedding_size))
        embedding = np.ctypeslib.as_ctypes(tmp)
        self.embedding = Array(embedding._type_, embedding, lock=False)

        # [V, senses , dim] 模型的多语境词向量对应的语境信息
        # 语境的第一个词向量为main-sense
        tmp = np.zeros(shape=(self.vocab_size, self.senses_number, self.embedding_size))
        senses = np.ctypeslib.as_ctypes(tmp)
        self.senses = Array(senses._type_, senses, lock=False)

        # [V, dim] 矩阵作为第二层权重
        tmp = np.zeros(shape=(self.vocab_size, self.embedding_size))
        weights = np.ctypeslib.as_ctypes(tmp)
        self.weights = Array(weights._type_, weights, lock=False)

    def getContextVector(self, context_ids):
        """通过上下文token的ID获取到上下文向量，并将上下文向量做average"""
        avg_vector = np.mean([self.main_embedding[t] for t in context_ids], axis=0)
        return avg_vector

    def getSimilarMax(self, context_vector, token):
        """将context vector与已经存在的sense比较，返回最相似的index，value"""
        current_count = self.senses_count[token]
        # candidate_vectors = np.insert(self.senses[token][0:current_count - 1], 0, values=self.main_sense[token], axis=0)
        candidate_vectors = np.insert(self.senses[token][0:current_count - 1], 0, values=self.main_sense[token], axis=0)
        cos_list = np.array([Util.cos_sim(context_vector, v) for v in candidate_vectors])
        cos_max_index = np.argmax(cos_list)
        cos_max_value = cos_list[cos_max_index]
        return cos_max_index, cos_max_value


    def updateEembedding(self, token_id, sense_index, gradient):
        """更新对应语境下的词向量 + 更新对应语境向量"""
        if sense_index == 0:
            self.main_embedding[token_id] += gradient
            self.main_sense[token_id] += gradient
        else:
            self.embedding[token_id][sense_index] += gradient
            self.senses[token_id][sense_index] += gradient

    def updateSense(self, token_id, sense_index, context_vector):
        """更新token对应的语境向量"""

    def updateWeights(self, token_id, gradient):
        self.weights[token_id] += gradient

    def saveEmbedding(self, epoch):
        """保存词向量，并将对应的语境向量也保存， 分开文件保存"""
        embedding_folder = self.args.embedding_folder
        print(" Saving {out_folder}".format(out_folder=embedding_folder))

        if not self.args.binary:
            embedding_path = os.path.join(embedding_folder, 'wv_epoch{epoch}.txt'.format(epoch=epoch))
            sense_path = os.path.join(embedding_folder, 'sv_epoch{epoch}.txt'.format(epoch=epoch))
            count_path = os.path.join(embedding_folder, 'count_epoch{epoch}.txt'.format(epoch=epoch))
            with open(embedding_path, 'w') as f_wv, open(sense_path, 'w') as f_sv, open(count_path, 'w') as f_count:
                f_wv.write("%d %d\n" % (len(self.embedding), self.args.embedding_size))
                f_sv.write("%d %d\n" % (len(self.embedding), self.args.embedding_size))
                f_count.write("%d %d\n" % (len(self.embedding), self.args.embedding_size))
                for index, vocab in tqdm.tqdm(enumerate(self.vocab)):
                    count = self.senses_count[index]
                    # 保存每个token的不同sense被count的次数
                    f_count.write("%s %d " % (vocab.word, count))
                    for value in self.senses_access[index][:count]:
                        f_count.write("%s " % value)
                    f_count.write("\n")

                    # 先把main 词向量和main 语境向量进行保存
                    f_wv.write("%s %d " % (vocab.word, count))
                    f_wv.write("%s " % " ".join([str(item) for item in self.main_embedding[index]]))

                    for vector in self.embedding[index][:count-1]:
                        f_wv.write("%s " % " ".join([str(item) for item in vector]))
                    f_wv.write("\n")

                    f_sv.write("%s %d " % (vocab.word, count))
                    f_sv.write("%s " % " ".join([str(item) for item in self.main_sense[index]]))

                    for vector in self.senses[index][:count-1]:
                        f_sv.write("%s " % " ".join([str(item) for item in vector]))
                    f_sv.write("\n")

