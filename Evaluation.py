#coding:utf-8

"""
衡量训练的词向量质量
"""

import sys
import gensim
from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format("out/training_201912" + sys.argv[1] + "/wv_epoch"+ sys.argv[2] +".txt", binary=False)
for word, sim in wv.most_similar('路线'):
    print("{word:>10s}: {sim:5.2f}".format(word=word, sim=sim))