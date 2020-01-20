#coding:utf-8

"""
衡量训练的词向量质量
"""

import sys, os
import tqdm
import numpy as np
import Util
import argparse
import multiprocessing

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
            jobs.append(multiprocessing.Process(target=self.worker, args=(thread_index, result_list, hit_list)))

        for job in jobs:
            job.start()
        for job in jobs:
            job.join()

        pred_list, real_list = list(), list()
        for (real, pred) in result_list:
            pred_list.append(pred)
            real_list.append(real)
        total = len(self.dataset.SCWS)
        real = len(pred_list)
        pearson = Util.Pearson(real_list, pred_list)
        print('{hit1}-{hit2}/{real}/{total} pearson:{p}'.format(hit1=hit_list[0], hit2=hit_list[1], real=real, total=total, p=pearson))

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

            for index, line in tqdm.tqdm(enumerate(lines[1:])):
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
        print("{real}/{total} pearson:{p}".format(real=real, total=total, p=pearson))

class MultiEmbedding:
    """多语境词向量"""

    def __init__(self):
        self.embedding = dict()
        self.senses = dict()
        self.senses_access = dict()
        self.senses_count = dict()
        self.vocab = list()

    def load(self, folder, epoch):
        """
        加载多语境词向量、多语境语境向量、词向量被access中的次数
        :param folder: traning文件夹下的目录
        :param epoch:  第几次循环
        :return:
        """
        self.wv_path = os.path.join(folder, "wv_epoch{epoch}.txt".format(epoch=epoch))
        self.sv_path = os.path.join(folder, "sv_epoch{epoch}.txt".format(epoch=epoch))
        self.count_path = os.path.join(folder, "count_epoch{epoch}.txt".format(epoch=epoch))

        with open(self.wv_path) as f:
            lines = f.readlines()
            vocab_size, embedding_size = lines[0].strip().split(" ", 1)
            vocab_size = int(vocab_size)
            self.vocab_size = vocab_size
            embedding_size = int(embedding_size)
            self.embedding_size = embedding_size

            for line in lines[1:]:
                word, sense_count, vectors = line.strip().split(" ", 2)
                sense_count = int(sense_count)
                self.vocab.append(word)
                self.senses_count[word] = int(sense_count)
                sense_vectors = np.array(vectors.split(" ")).astype(float).reshape((sense_count, embedding_size))
                for index in range(sense_count):
                    vector = sense_vectors[index]
                    self.embedding['%s_%d' % (word, index + 1)] = vector

        with open(self.sv_path) as f:
            lines = f.readlines()
            for line in lines[1:]:
                word, sense_count, vectors = line.strip().split(" ", 2)
                sense_count = int(sense_count)
                sense_vectors = np.array(vectors.split(" ")).astype(float).reshape((sense_count, embedding_size))
                self.senses[word] = sense_vectors

        with open(self.count_path) as f:
            lines = f.readlines()
            for line in lines[1:]:
                word, sense_count, vector = line.strip().split(" ", 2)
                access_vector = np.array(vector.split(" ")).astype(int)
                self.senses_access[word] = access_vector


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

    def get_context_embedding(self, sentences):
        """在scws评测中，获取context embedding，并得到最对应的sense的索引"""
        # 测评数据集中有的有多个word，或者是有的<b> </b>中间的单词和给定的单词大小写不一致
        left_context = sentences.split("<b>")[0]
        right_context = sentences.split("</b>")[1]
        left_context_embedding = [self.embedding['{0}_1'.format(word)] for word in left_context.split(" ") if word in self.vocab]
        right_context_embedding = [self.embedding['{0}_1'.format(word)] for word in right_context.split(" ") if word in self.vocab]
        context_embedding = np.mean(left_context_embedding[0:5] + right_context_embedding[0:5], axis=0)
        return context_embedding

    def get_sim_sense_embedding(self, context_embedding, word):
        """"""
        use_sense_embedding = False
        cos_sim_list = [Util.cos_sim(context_embedding, vector) for vector in self.senses[word]]
        cos_max_index = np.argmax(cos_sim_list)
        cos_max_value = cos_sim_list[cos_max_index]
        if cos_max_value < 0.5:
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
                return main_access*(main_access/access_total) + sense_embedding*(sense_access/access_total), use_sense_embedding


    def evalSCWS(self, scws_path, use_main):
        """
        多语境词向量的scws评测
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
        print('{hit1}-{hit2}/{real}/{total} pearson:{p}'.format(hit1=self.hit1, hit2=self.hit2, real=real, total=total, p=pearson))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-single", dest='single', required=True, type=int, help="是评价单语境词向量，还是多语境词向量（0=single， 1=multiple）")
    parser.add_argument("-day", dest='day', required=True, type=str, help="词向量的文件路径中的天数")
    parser.add_argument("-time", dest='time', required=True, type=str, help="词向量路径中的时间戳")
    parser.add_argument("-epoch", dest='epoch', required=True, type=str, help="训练词向量的次数")
    parser.add_argument("-threads", dest='threads', required=True, type=int, help="评测SCWS时候用多线程")
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

        embedding = MultiEmbedding()
        embedding.load('out/training_2020{0}-{1}'.format(args.day, args.time), args.epoch)
        print("Multiple embedding loading success.")
        # embedding.evalWS353('./data/ws/ws353.txt', use_avg=False)
        # embedding.evalWS353('./data/ws/ws353.txt', use_avg=True)
        # embedding.evalSCWS('./data/scws/ratings.txt', use_main=True)
        # embedding.evalSCWS('./data/scws/ratings.txt', use_main=False)

    dataset = Dataset()
    # embedding = MultiEmbedding()
    # embedding.load('out/training_2020{0}-{1}'.format(args.day, args.time), args.epoch)

    eval = Evaluation(dataset, embedding, args.threads, use_context=True)
