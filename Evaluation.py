#coding:utf-8

"""
衡量训练的词向量质量
"""

import sys, os
import gensim
from gensim.models import KeyedVectors
import tqdm
import numpy as np
import Util

folder_path = sys.argv[1]
epoch_index = sys.argv[2]
#
#
# wv = KeyedVectors.load_word2vec_format("out/training_2020" + folder_part + "/wv_epoch"+ epoch_index +".txt", binary=False)
# for word, sim in wv.most_similar('风景'):
#     print("{word:>10s}: {sim:5.2f}".format(word=word, sim=sim))


class SingleEmbedding:
    """单一语境词向量"""

    def __init__(self, path):
        self.path = path
        self.embedding = dict()

    def load(self):
        pass


class MultiEmbedding:
    """多语境词向量"""

    def __init__(self, folder, epoch):
        self.wv_path = os.path.join(folder, "wv_epoch{epoch}.txt".format(epoch=epoch))
        self.sv_path = os.path.join(folder, "sv_epoch{epoch}.txt".format(epoch=epoch))
        self.count_path = os.path.join(folder, "count_epoch{epoch}.txt".format(epoch=epoch))
        self.embedding = dict()
        self.sense = dict()
        self.count = dict()

    def load(self):
        print("Load multi embedding from %s" % (self.wv_path))
        with open(self.wv_path) as f:
            lines = f.readlines()
            vocab_size, embedding_size = lines[0].strip().split(" ", 1)
            self.vocab = list()
            vocab_size = int(vocab_size)
            embedding_size = int(embedding_size)
            self.sense_count = dict()

            for index, line in tqdm.tqdm(enumerate(lines[1:])):
                word, sense_count, vector = line.strip().split(" ", 2)
                self.vocab.append(word)
                sense_count = int(sense_count)
                self.sense_count[word] = sense_count
                if sense_count > 11:
                    print(word, sense_count)
                    continue
                sense_vectors = np.array(vector.split(" ")).astype(float).reshape((sense_count, embedding_size))
                for sense_index in range(sense_count):
                    sense_vector = sense_vectors[sense_index]
                    self.embedding['%s_%d' % (word, sense_index+1)] = sense_vector

        print("Load multi sense from %s" % (self.sv_path))
        with open(self.sv_path) as f:
            lines = f.readlines()
            vocab_size, embedding_size = lines[0].strip().split(" ", 1)

        print("Load sense access from %s" % (self.count_path))
        with open(self.count_path) as f:
            lines = f.readlines()
            vocab_size, embedding_size = lines[0].strip().split(" ", 1)
            vocab_size = int(vocab_size)
            embedding_size = int(embedding_size)
            self.senses_access = dict()

            for index, line in tqdm.tqdm(enumerate(lines[1:])):
                word, sense_count, vector = line.strip().split(" ", 2)
                sense_count = int(sense_count)
                if sense_count > 11:
                    print(word, sense_count)
                    continue
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

multiEmbedding = MultiEmbedding('out/training_20200109-{0}'.format(folder_path), epoch_index)
multiEmbedding.load()
while True:
    word = input("Please enter a word [X quit]")
    if 'X' == word:
        exit()
    else:
        multiEmbedding.most_similar(word)