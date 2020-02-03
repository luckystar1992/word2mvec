#coding:utf-8

"""
衡量训练的词向量质量
加载词向量的方式一共有两种，一种是单线程加载技术，Multiple load1，一种是多线程加载 load2
衡量scws的方式也是有两种，一种是利用Manager共享变量，一种是利用Array共享内存
"""

import sys, os
import time
import tqdm
import numpy as np
import Util
import argparse
import multiprocessing
from multiprocessing import Array, Value, Pool
import numpy as np
import warnings
from collections import Counter

WS353_Path = 'data/ws/ws353.txt'
WS353_Rela_Path = 'data/ws/ws353_relatedness.txt'
WS353_Sim_Path = 'data/ws/ws353_similarity.txt'
SCWS_Path = 'data/scws/ratings.txt'

class Dataset:

    def __init__(self):
        # 单一语境下使用的数据集
        self.WS353 = list()
        self.WS353_Rela = list()
        self.WS353_Sim = list()
        # SCWS数据集，在单一语境下和上下文语境下
        self.SCWS = list()

        # 直接加载各项数据集
        self.loadWS353()
        self.loadSCWS()
        self.loadWS353_rela()
        self.loadWS353_sim()

    def loadWS353(self):
        """加载ws353数据集"""
        with open(WS353_Path) as f:
            for line in f.readlines():
                _word1, _word2, _score = line.strip().split("\t")
                self.WS353.append((_word1, _word2, _score))

    def loadWS353_rela(self):
        """加载ws353相关性的数据集"""
        with open(WS353_Rela_Path) as f:
            for line in f.readlines():
                _word1, _word2, _score = line.strip().split("\t")
                self.WS353_Rela.append((_word1, _word2, _score))

    def loadWS353_sim(self):
        with open(WS353_Sim_Path) as f:
            for line in f.readlines():
                _word1, _word2, _score = line.strip().split("\t")
                self.WS353_Sim.append((_word1, _word2, _score))

    def loadSCWS(self):
        with open(SCWS_Path) as f:
            for line in f.readlines():
                _index, _word1, _pos1, _word2, _pos2, _sen1, _sen2, _score, *_scores = line.strip().split("\t")
                assert len(_scores) == 10
                self.SCWS.append((_word1, _pos1, _word2, _pos2, _sen1, _sen2, _score))

class Evaluation:
    """使用多线程评测SCWS的加速"""
    def __init__(self, dataset, embedding, threads, use_context):
        self.dataset = dataset
        self.embedding = embedding
        self.use_context = use_context

        scws_len = len(self.dataset.SCWS)
        interval = int(scws_len/threads)
        sub_index = range(0, scws_len + interval, interval)
        self.start_list = sub_index[0:-1]
        self.end_list = sub_index[1:]
        self.lock = multiprocessing.Lock()

        manager = multiprocessing.Manager()
        result_list = manager.list()
        hit_list = manager.list((0, 0))
        jobs = list()
        for thread_index in range(0, len(self.start_list)):
            job = multiprocessing.Process(target=self.worker, args=(thread_index, result_list, hit_list))
            job.start()
            jobs.append(job)
        for thread_index in range(0, len(self.start_list)):
            jobs[thread_index].join()

        pred_list, real_list = list(), list()
        for (real, pred) in result_list:
            pred_list.append(pred)
            real_list.append(real)
        total = len(self.dataset.SCWS)
        real = len(pred_list)
        pearson = Util.Pearson(real_list, pred_list)
        self.pearson = pearson
        print('h1:{hit1:>4d} h2:{hit2>4d} {real:>4d}/{total} pearson:{p:>6.4f}'.format(hit1=hit_list[0], hit2=hit_list[1], real=real, total=total, p=pearson))

    def getSubSCWS(self, thread_index):
        """获取子子线程需要处理的SCWS行数"""
        start = self.start_list[thread_index]
        end = self.end_list[thread_index]
        return self.dataset.SCWS[start:end]

    def worker(self, thread_index, result_list, hit_list):
        """实际的工作子线程"""
        sub_scws_list = self.getSubSCWS(thread_index)
        for (_word1, _pos1, _word2, _pos2, _sen1, _sen2, _score) in sub_scws_list:
            if _word1 in self.embedding.vocab and _word2 in self.embedding.vocab:
                if self.use_context:
                    context_embedding1 = self.embedding.get_context_embedding(_sen1)
                    context_embedding2 = self.embedding.get_context_embedding(_sen2)
                    vector1, use_sense_embedding = self.embedding.get_sim_sense_embedding(context_embedding1, _word1)
                    if use_sense_embedding:
                        # 注意这里使用进程同步锁，防止出现计数错误的现象
                        self.lock.acquire()
                        hit_list[0] += 1
                        self.lock.release()
                    vector2, use_sense_embedding = self.embedding.get_sim_sense_embedding(context_embedding2, _word2)
                    if use_sense_embedding:
                        self.lock.acquire()
                        hit_list[1] += 1
                        self.lock.release()
                else:
                    vector1 = self.embedding.embedding["{word}_1".format(word=_word1)]
                    vector2 = self.embedding.embedding["{word}_1".format(word=_word2)]
                pred_sim = Util.cos_sim(vector1, vector2)
                # 在这里append的时候一定要将两者同时append到list中去，如果是两个不同的
                # list的话，多线程会出现错位的现象
                result_list.append((pred_sim, float(_score)))


class Evaluation2:

    def __init__(self, args, dataset, embedding, use_context):
        self.args = args
        self.dataset = dataset
        self.embedding = embedding
        self.use_context = use_context
        self.real = 0
        self.total = len(self.embedding.vocab)
        self.realSCWS = list()

        # 先排查哪些scws的word1和word2不在embedding中
        for sample in self.dataset.SCWS:
            if sample[0] in self.embedding.vocab and sample[2] in self.embedding.vocab:
                self.realSCWS.append(sample)

        # 计算每个线程处理的有效scws的行起终点
        interval = int(np.ceil(len(self.realSCWS)/args.threads))
        line_index = range(0, len(self.realSCWS) + interval, interval)

        # 更新参数项并将需要统计的word1和word2的命中统计数目更新到args中
        self.args.start_list = line_index[0:-1]
        self.args.end_list = line_index[1:]
        self.args.use_context = use_context

    def eval(self, sim_type):
        """实现Huang2012提出的4种不同相似度的评测方法"""
        # 建立共享内存用于多线程worker的共享使用  两个列向量，real sim 和 pred sim 两个相似度
        assert sim_type in ['global', 'average', 'averagec', 'local']
        tmp = np.zeros(shape=(len(self.realSCWS), 2))
        simArray = np.ctypeslib.as_ctypes(tmp)
        simArray = Array(simArray._type_, simArray, lock=False)

        word1_hit, word2_hit = Value('i', 0), Value('i', 0)
        self.args.word1_hit = word1_hit
        self.args.word2_hit = word2_hit
        self.args.sim_type = sim_type

        pool = Pool(processes=self.args.threads,
                    initializer=Evaluation2.init_worker,
                    initargs=(self.args, self.embedding, self.realSCWS, simArray))
        pool.map(Evaluation2.run_worker, range(0, self.args.threads))

        pred_list = [item[0] for item in simArray]
        real_list = [item[1] for item in simArray]
        # 最后统计
        hit1 = args.word1_hit.value
        hit2 = args.word2_hit.value
        real = len(self.realSCWS)
        total = len(self.dataset.SCWS)
        pearson = Util.Pearson(pred_list, real_list)
        self.pearson = pearson
        print('h1:{hit1:>4d} h2:{hit2:>4d} {real:>4d}/{total} pearson:{p:>6.4f}({sim_type:>8s})'.format(sim_type=sim_type, hit1=hit1, hit2=hit2, real=real,
                                                                total=total, p=pearson))

    @classmethod
    def init_worker(cls, *params):
        global args, embedding, scws, simArray
        args, embedding, scws, simArray = params

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            simArray = np.ctypeslib.as_array(simArray)

    @classmethod
    def run_worker(cls, thread_index):
        start = args.start_list[thread_index]
        end = args.end_list[thread_index]
        for index, (_word1, _pos1, _word2, _pos2, _sen1, _sen2, _score) in enumerate(scws[start: end]):
            if args.sim_type == 'averagec':
                global_sim = 0
                word1_access = embedding.senses_access[_word1]
                word2_access = embedding.senses_access[_word2]
                word1_access_total = sum(word1_access)
                word2_access_total = sum(word2_access)
                for index1, access1 in enumerate(word1_access):
                    p1 = access1 / word1_access_total
                    embedd1 = embedding.embedding["{word}_{index}".format(word=_word1, index=index1 + 1)]
                    for index2, access2 in enumerate(word2_access):
                        p2 = access2 / word2_access_total
                        embedd2 = embedding.embedding["{word}_{index}".format(word=_word2, index=index2+1)]
                        global_sim += p1 * p2 * Util.cos_sim(embedd1, embedd2)
                #global_sim /= word1_count * word2_count
                simArray[start+index][0] = global_sim
                simArray[start+index][1] = float(_score)

            elif args.sim_type == 'average':
                global_sim = 0
                word1_count = embedding.senses_count[_word1]
                word2_count = embedding.senses_count[_word2]
                for index1 in range(0, word1_count):
                    embedd1 = embedding.embedding["{word}_{index}".format(word=_word1, index=index1+1)]
                    for index2 in range(0, word2_count):
                        embedd2 = embedding.embedding["{word}_{index}".format(word=_word2, index=index2+1)]
                        global_sim += Util.cos_sim(embedd1, embedd2)
                global_sim /= word1_count * word2_count
                simArray[start + index][0] = global_sim
                simArray[start + index][1] = float(_score)

            elif args.sim_type == 'global':
                vector1 = embedding.embedding["{word}_1".format(word=_word1)]
                vector2 = embedding.embedding["{word}_1".format(word=_word2)]
                simArray[start+index][0] = Util.cos_sim(vector1, vector2)
                simArray[start+index][1] = float(_score)

            # 这个应该是最复杂的一个评测方法，需要通过word的context embedding进行确切word embedding的查找
            # 最后通过确切查找出来的两个index进行 cos的确定
            elif args.sim_type == 'local':
                context_embedding1 = embedding.get_context_embedding(_sen1, args.window_size)
                context_embedding2 = embedding.get_context_embedding(_sen2, args.window_size)
                vector1, use_sense_embedding = embedding.get_sim_sense_embedding(context_embedding1, _word1)
                if use_sense_embedding:
                    args.word1_hit.value += 1
                vector2, use_sense_embedding = embedding.get_sim_sense_embedding(context_embedding2, _word2)
                if use_sense_embedding:
                    args.word2_hit.value += 1
                local_sim = Util.cos_sim(vector1, vector2)
                simArray[start + index][0] = local_sim
                simArray[start + index][1] = float(_score)
            elif args.sim_type == 'local1':
                pass
            else:
                raise NameError

            # if args.use_context:
            #     context_embedding1 = embedding.get_context_embedding(_sen1)
            #     context_embedding2 = embedding.get_context_embedding(_sen2)
            #     vector1, use_sense_embedding = embedding.get_sim_sense_embedding(context_embedding1, _word1)
            #     if use_sense_embedding:
            #         args.word1_hit.value += 1
            #     vector2, use_sense_embedding = embedding.get_sim_sense_embedding(context_embedding2, _word2)
            #     if use_sense_embedding:
            #         args.word2_hit.value += 1
            # else:
            #     vector1 = embedding.embedding["{word}_1".format(word=_word1)]
            #     vector2 = embedding.embedding["{word}_1".format(word=_word2)]
            # pred_sim = Util.cos_sim(vector1, vector2)
            # # 在这里append的时候一定要将两者同时append到list中去，如果是两个不同的
            # # list的话，多线程会出现错位的现象
            # simArray[start+index][0] = pred_sim
            # simArray[start+index][1] = float(_score)

    def count_multi_hit(self):
        """
        统计scws的有效例子中有多少个是多义词，同时统计word1和word2
        :return:
        """
        s1, s2, s3 = 0, 0, 0
        for sample in self.realSCWS:
            word1 = sample[0]
            word2 = sample[2]
            if self.embedding.senses_count[word1] > 1:
                s1 += 1
            if self.embedding.senses_count[word2] > 1:
                s2 += 1
            if self.embedding.senses_count[word1] > 1 or self.embedding.senses_count[word2] > 1:
                s3 += 1
        print("s1:{s1:>4d} s2:{s2:>4d} {s3:>4d}/{s4}".format(s1=s1, s2=s2, s3=s3, s4=len(self.realSCWS)))


class SingleEmbedding:
    """单一语境词向量"""

    def __init__(self):
        self.embedding = dict()


    def load(self, wv_path):
        """加载单一语境词向量"""
        # self.wv_path = os.path.join(folder, 'wv_epoch{0}.txt'.format(epoch))
        self.wv_path = wv_path
        with open(self.wv_path) as f:
            lines = f.readlines()
            vocab_size, embedding_size = lines[0].strip().split(" ", 1)
            vocab_size = int(vocab_size)
            embedding_size = int(embedding_size)
            self.vocab = list()

            for index, line in enumerate(lines[1:]):
                word, vector = line.strip().split(" ", 1)
                self.vocab.append(word)
                vector = np.array(vector.split(" ")).astype(float)
                self.embedding[word] = vector

    def most_similar(self, word, top_n=5):
        """单一语境词向量进行top的返回"""
        if word not in self.vocab:
            print("{0} not included in the Embedding".format(word))
        else:
            word_vector = self.embedding[word]
            cos_dict = dict()
            for key, vector in self.embedding.items():
                if key is not word:
                    cos_dict[key] = Util.cos_sim(word_vector, vector)
            cos_dict = sorted(cos_dict.items(), key=lambda x: x[1], reverse=True)
            print("Top-{0} most similar of {1}".format(top_n, word))
            for key, cos in cos_dict[:top_n]:
                print("{0:\u3000<10}  {1:>5.3f}".format(key, cos))

    def evalWS353(self, ws353_path):
        """ WS353数据集的评测
        :param ws353_path:
        :return:
        """
        total, real = 0, 0
        real_sim_list, pred_sim_list = list(), list()
        with open(ws353_path) as f:
            for line in f.readlines():
                total += 1
                word_a, word_b, sim = line.strip().split("\t")
                if word_a in self.vocab and word_b in self.vocab:
                    real += 1
                    vector_a = self.embedding[word_a]
                    vector_b = self.embedding[word_b]
                    pred_sim = Util.cos_sim(vector_a, vector_b)
                    real_sim_list.append(float(sim))
                    pred_sim_list.append(pred_sim)
        pearson = Util.Pearson(pred_sim_list, real_sim_list)
        self.pearson_ws353 = pearson
        print('{real}/{total} pearson:{p}'.format(real=real, total=total, p=pearson))

    def evalSCWS(self, scws_path):
        total, real = 0, 0
        real_sim_list, pred_sim_list = list(), list()
        with open(scws_path) as f:
            for line in f.readlines():
                total += 1
                _index, _word1, _pos1, _word2, _pos2, _sen1, _sen2, _score, *_scores = line.strip().split("\t")
                if _word1 in self.vocab and _word2 in self.vocab:
                    real += 1
                    vector_1 = self.embedding[_word1]
                    vector_2 = self.embedding[_word2]
                    real_sim_list.append(float(_score))
                    pred_sim_list.append(Util.cos_sim(vector_1, vector_2))
        pearson = Util.Pearson(pred_sim_list, real_sim_list)
        self.pearson_scws = pearson
        print("{real}/{total} pearson:{p}".format(real=real, total=total, p=pearson))


class MultiEmbedding:
    """
        多语境词向量的加载评测
    """

    def __init__(self, args):
        self.args = args
        self.embedding = dict()
        self.senses = dict()
        self.senses_access = dict()
        self.senses_count = dict()
        self.vocab = list()

        folder = os.path.join('out/training_2020{day}-{time}'.format(day=args.day, time=args.time))
        epoch = args.epoch
        self.wv_path = os.path.join(folder, "wv_epoch{epoch}.txt".format(epoch=epoch))
        self.sv_path = os.path.join(folder, "sv_epoch{epoch}.txt".format(epoch=epoch))
        self.count_path = os.path.join(folder, "count_epoch{epoch}.txt".format(epoch=epoch))

        # 读取单词列表，senses计数和access计数
        with open(self.count_path) as f:
            lines = f.readlines()
            _vocab_size, _embedding_size = lines[0].strip().split(" ", 1)
            self.vocab_size = int(_vocab_size)
            self.embedding_size = int(_embedding_size)
            for line in lines[1:]:
                word, sense_count, vector = line.strip().split(" ", 2)
                self.vocab.append(word)
                self.senses_count[word] = int(sense_count)
                access_vector = np.array(vector.split(" ")).astype(int)
                self.senses_access[word] = access_vector

    def load1(self):
        """
            使用单线程技术加载词向量
        """
        with open(self.wv_path) as f:
            lines = f.readlines()
            for line in tqdm.tqdm(lines[1:]):
                word, sense_count, vectors = line.strip().split(" ", 2)
                sense_count = int(sense_count)
                sense_vectors = np.array(vectors.split(" ")).astype(float).reshape((self.senses_count[word], self.embedding_size))
                for index in range(sense_count):
                    vector = sense_vectors[index]
                    self.embedding['%s_%d' % (word, index + 1)] = vector

        with open(self.sv_path) as f:
            lines = f.readlines()
            for line in tqdm.tqdm(lines[1:]):
                word, sense_count, vectors = line.strip().split(" ", 2)
                sense_vectors = np.array(vectors.split(" ")).astype(float).reshape((self.senses_count[word], self.embedding_size))
                self.senses[word] = sense_vectors

    def load2(self):
        """
        使用多线程技术加载
        """
        # 利用multiprocessing的Array进行并行化处理
        # 这里为了简便操作，将每个word的sense数目预先增加到固定数目，最终使用的时候再通过实际的count取用
        max_count = max(self.senses_count.values())
        tmp = np.zeros(shape=(self.vocab_size, max_count, self.embedding_size), dtype='float32')
        embedding = np.ctypeslib.as_ctypes(tmp)
        embeddingArray = Array(embedding._type_, embedding, lock=False)

        tmp = np.zeros(shape=(self.vocab_size, max_count, self.embedding_size), dtype='float32')
        senses = np.ctypeslib.as_ctypes(tmp)
        sensesArray = Array(senses._type_, senses, lock=False)

        with open(self.wv_path) as f_wv, open(self.sv_path) as f_sv:
            wvlines = f_wv.readlines()
            svlines = f_sv.readlines()
            wvlines = wvlines[1:]
            svlines = svlines[1:]

        # 处理各个子线程读取的文件开始和结尾并将这些参数放到args中
        interval = int(self.vocab_size / self.args.threads)
        line_index = range(0, self.vocab_size + interval, interval)
        args.start_list = line_index[0:-1]
        args.end_list = line_index[1:]
        args.embedding_size = self.embedding_size
        pool = Pool(processes=self.args.threads,
                    initializer=MultiEmbedding.init_worker,
                    initargs=(wvlines, svlines, embeddingArray, sensesArray, args))
        pool.map(MultiEmbedding.run_worker, range(0, args.threads))

        embeddingArray = np.array(embeddingArray)
        sensesArray = np.array(sensesArray)

        for word_index, (word, wv_vectors, sv_vectors) in enumerate(zip(self.vocab, embeddingArray, sensesArray)):
            count = self.senses_count[word]
            sense_list = list()
            for count_index in range(0, count):
                self.embedding['{0}_{1}'.format(word, count_index+1)] = embeddingArray[word_index][count_index]
                sense_list.append(sensesArray[word_index][count_index])
            self.senses['{0}'.format(word)] = np.array(sense_list)

    @classmethod
    def init_worker(cls, *params):
        """MultiEmbedding load2方法中的子线程初始化方法"""
        global wvlines, svlines, embeddingArray, sensesArray, args
        wvlines, svlines, embeddingArray, sensesArray, args = params

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            embeddingArray = np.ctypeslib.as_array(embeddingArray)
            sensesArray = np.ctypeslib.as_array(sensesArray)

    @classmethod
    def run_worker(self, thread_index):
        """MultiEmbedding load2方法中的子线程的工作函数"""
        start = args.start_list[thread_index]
        end = args.end_list[thread_index]
        worker_lines = wvlines[start:end]
        for line_index, line in enumerate(worker_lines):
            word, sense_count, vectors = line.strip().split(" ", 2)
            sense_count = int(sense_count)
            sense_vectors = np.array(vectors.split(" ")).astype(float).reshape((sense_count, args.embedding_size))
            array_index = start + line_index
            for sense_index, vector in enumerate(sense_vectors):
                embeddingArray[array_index][sense_index] = vector
        worker_lines = svlines[start:end]
        for line_index, line in enumerate(worker_lines):
            word, sense_count, vectors = line.strip().split(" ", 2)
            sense_count = int(sense_count)
            sense_vectors = np.array(vectors.split(" ")).astype(float).reshape((sense_count, args.embedding_size))
            array_index = start + line_index
            for sense_index, vector in enumerate(sense_vectors):
                sensesArray[array_index][sense_index] = vector

    def count_statistic(self):
        """将不同sense出现次数进行统计"""
        counter = Counter(self.senses_count.values())
        print("Count statistic of {path}".format(path=self.count_path))
        for key, value in sorted(counter.most_common(), key=lambda x: x[0]):
            print("{key:>2d}: {value1:>6d} {value2:>6.2f}% ".format(key=key, value1=value, value2=value*100/len(self.senses_count)))

    def most_similar(self, word, top_n=5):
        if word not in self.vocab:
            print("%s not included in the Embedding" % word)
        else:
            sense_count = self.sense_count[word]
            for sense_index in range(sense_count):
                word_vector = self.embedding['%s_%d' % (word, sense_index+1)]
                cos_dict = dict()
                for key, vector in self.embedding.items():
                    cos_dict[key] = Util.cos_sim(word_vector, vector)
                cos_dict = sorted(cos_dict.items(), key=lambda x: x[1], reverse=True)
                print("Top-%d most similar of %s_%d (access %d)" % (top_n, word, sense_index+1, self.senses_access[word][sense_index]))
                for key, cos in cos_dict[1:top_n+1]:
                    print("{0:^10}  {1:5.3f}".format(key, cos, chr(12288)))

    def get_avg_embedding(self, word):
        all_access = sum(self.senses_access[word])
        embedding = np.zeros(self.embedding_size)
        for index, access in enumerate(self.senses_access[word]):
            embedding += self.embedding["{word}_{index}".format(word=word, index=index+1)] * (access/all_access)
        return embedding

    def evalWS353(self, ws353_path, use_avg):
        """
        多语境词向量的ws353评测，这里只使用第一个embeding
        :param ws353_path: ws353的路径
        :param use_avg:    使用main-embedding作为词向量，还是使用所有词向量的平均作为
        :return: 返回皮尔逊系数
        """
        total, real = 0, 0
        real_sim_list, pred_sim_list = list(), list()
        with open(ws353_path) as f:
            for line in f.readlines():
                total += 1
                word_a, word_b, sim = line.strip().split("\t")
                if word_a in self.vocab and word_b in self.vocab:
                    real += 1
                    if use_avg:
                        vector_a = self.get_avg_embedding(word_a)
                        vector_b = self.get_avg_embedding(word_b)
                    else:
                        word_a = word_a + "_1"
                        word_b = word_b + "_1"
                        vector_a = self.embedding[word_a]
                        vector_b = self.embedding[word_b]
                    pred_sim = Util.cos_sim(vector_a, vector_b)
                    real_sim_list.append(float(sim))
                    pred_sim_list.append(pred_sim)
        pearson = Util.Pearson(pred_sim_list, real_sim_list)
        print('{real}/{total} pearson:{p}'.format(real=real, total=total, p=pearson))

    def get_context_embedding(self, sentences, window_size):
        """在scws评测中，获取context embedding，并得到最对应的sense的索引"""
        # 测评数据集中有的有多个word，或者是有的<b> </b>中间的单词和给定的单词大小写不一致
        left_context = sentences.split("<b>")[0]
        right_context = sentences.split("</b>")[1]
        left_context_embedding = [self.embedding['{0}_1'.format(word)] for word in left_context.split(" ")[::-1] if word in self.vocab]
        right_context_embedding = [self.embedding['{0}_1'.format(word)] for word in right_context.split(" ") if word in self.vocab]
        context_embedding = np.mean(left_context_embedding[0:window_size] + right_context_embedding[0:window_size], axis=0)
        return context_embedding

    def get_sim_sense_embedding(self, context_embedding, word):
        """"""
        cos_sim_list = [Util.cos_sim(context_embedding, vector) for vector in self.senses[word]]
        cos_max_index = np.argmax(cos_sim_list)
        cos_max_value = cos_sim_list[cos_max_index]
        if cos_max_value < args.sim_threshold:
            use_sense_embedding = False
            return self.embedding['{0}_1'.format(word)], use_sense_embedding
        else:
            if cos_max_index == 0:
                use_sense_embedding = False
                return self.embedding['{0}_1'.format(word)], use_sense_embedding
            else:
                use_sense_embedding = True
                main_embedding = self.embedding['{0}_1'.format(word)]
                sense_embedding = self.embedding['{0}_{1}'.format(word, cos_max_index + 1)]
                access_total = sum(self.senses_access[word])
                main_access = self.senses_access[word][0]
                sense_access = self.senses_access[word][cos_max_index]
                # 如果main的系数是-1的话，那么就说明使用的是main和sense按照access混合的策略
                if args.main_alpha == -1:
                    return main_access * (main_embedding / access_total) + sense_embedding * (
                                sense_access / access_total), use_sense_embedding
                # 如果main_alpha范围为(0，1]的话，就说明使用的是main和sense按照既定的混合策略
                else:
                    return main_embedding * args.main_alpha + sense_embedding * (1 - args.main_alpha), use_sense_embedding


    def evalSCWS(self, scws_path, use_main):
        """
        多语境词向量的scws评测，这是单线程的评测方法，使用的时候请使用多线程的Evaluation类
        :param scws_path:
        :param use_main:  所有的多语境词向量使用第一个main词向量
        :return:
        """
        total, real = 0, 0
        real_sim_list, pred_sim_list = list(), list()
        self.hit1, self.hit2 = 0, 0
        with open(scws_path) as f:
            for line in f.readlines():
                total += 1
                _index, _word1, _pos1, _word2, _pos2, _sen1, _sen2, _score, *_scores = line.strip().split("\t")
                if _word1 in self.vocab and _word2 in self.vocab:
                    real += 1
                    if use_main:
                        vector_1 = self.embedding['{0}_1'.format(_word1)]
                        vector_2 = self.embedding['{0}_1'.format(_word2)]
                    else:
                        context_embedding1 = self.get_context_embedding(_sen1)
                        context_embedding2 = self.get_context_embedding(_sen2)
                        vector_1, use_sense_embedding = self.get_sim_sense_embedding(context_embedding1, _word1)
                        if use_sense_embedding:
                            self.hit1 += 1
                        vector_2, use_sense_embedding = self.get_sim_sense_embedding(context_embedding2, _word2)
                        if use_sense_embedding:
                            self.hit2 += 1
                    real_sim_list.append(float(_score))
                    pred_sim_list.append(Util.cos_sim(vector_1, vector_2))

        pearson = Util.Pearson(pred_sim_list, real_sim_list)
        print('{hit1}-{hit2}/{real}/{total} pearson:{p:>6.4f}'.format(hit1=self.hit1, hit2=self.hit2, real=real, total=total, p=pearson))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-single", dest='single', required=True, type=int, help="是评价单语境词向量，还是多语境词向量（0=single， 1=multiple）")
    parser.add_argument("-day", dest='day', required=True, type=str, help="词向量的文件路径中的天数")
    parser.add_argument("-time", dest='time', required=True, type=str, help="词向量路径中的时间戳")
    parser.add_argument("-epoch", dest='epoch', required=True, type=str, help="训练词向量的次数")
    parser.add_argument("-threads", dest='threads', required=True, type=int, help="评测SCWS时候用多线程")
    parser.add_argument("-sim_threshold", dest='sim_threshold', type=float, help="sense查找时候使用的相似度")
    parser.add_argument("-main_alpha", dest='main_alpha', type=float, help="main和sense使用的时候的系数")
    parser.add_argument("-window_size", dest="window_size", type=int, help="寻找上下文的窗口大学")
    args = parser.parse_args()

    ws353 = './data/ws/ws353.txt'
    scws = './data/scws/ratings.txt'
    # 如果是单一词向量语境模型的话
    if args.single:
        embedding = SingleEmbedding()
        wv_path = 'out/training_2020{0}-{1}/wv_epoch{2}.txt'.format(args.day, args.time, args.epoch)
        embedding.load(wv_path)
        embedding.evalWS353(ws353)
        embedding.evalSCWS(scws)
    else:
        t_begin = time.time()
        embedding = MultiEmbedding(args)
        embedding.load2()
        embedding.count_statistic()
        dataset = Dataset()
        # 说明只使用main
        if args.main_alpha == 1:
            eval = Evaluation2(args, dataset, embedding, False)
            eval.count_multi_hit()
            eval.eval('global')
            eval.eval('average')
            eval.eval('averagec')
            eval.eval('local')
        else:
            eval2 = Evaluation2(args, dataset, embedding, True)
            eval2.count_multi_hit()
            eval2.eval('global')
            eval2.eval('average')
            eval2.eval('averagec')
            eval2.eval('local')
        t_end = time.time()

