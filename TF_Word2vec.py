#coding:utf-8
"""
Tensorflow2 word2vec
"""

import sys, os
import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from FileUtil import Vocab, DataSet
import Util
import argparse
import numpy as np
import math

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def dataProcess(args, vocab):
    """ 处理输入文本，转换成TF2可用的格式 """
    train_x, train_y = list(), list()
    window_size = args.window_size
    with open(args.input) as f_input:
        for line in tqdm.tqdm(f_input.readlines()):
            tokens = line.strip().split()
            tokens_indices = vocab.indices(tokens)
            for index, target_word in enumerate(tokens_indices):
                context_words = list()
                begin = index - window_size if index - window_size > 0 else 0
                end = index + 1 + window_size if index + window_size + 1 < len(tokens_indices) else len(tokens_indices)
                context_words.extend(tokens_indices[begin:index])
                context_words.extend(tokens_indices[index+1:end])
                if args.cbow > 0:
                    train_x.append(context_words)
                    train_y.append(target_word)
                else:
                    train_x.extend([target_word] * len(context_words))
                    train_y.extend(context_words)

    return np.array(train_x), np.array(train_y)



class Word2vec(keras.Model):

    def __init__(self, args, vocab):
        super(Word2vec, self).__init__()
        self.args = args
        self.vocab = vocab
        self.embeddings = layers.Embedding(len(self.vocab), self.args.embedding_size)
        self.nce_weights = tf.Variable(tf.random.truncated_normal(
            [len(self.vocab), self.args.embedding_size],
            stddev=1.0/math.sqrt(self.args.embedding_size)
        ))
        self.nce_biases = tf.Variable(tf.zeros([len(self.vocab)]))

    def call(self, inputs, training=None, mask=None):
        embed = self.embeddings(inputs)
        return embed, self.nce_weights, self.nce_biases


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('-input', dest='input', required=True, help='训练语料的路径')
    parse.add_argument('-out_folder', dest='out_folder', default='./out_tf2', help='模型向量的保存文件夹')
    parse.add_argument('-vocab_size', dest='vocab_size', required=True, type=int, help='词典能保存的最大单词数目')
    parse.add_argument('-cbow', dest='cbow', required=True, type=int, help='采用cbow模型还是skip-gram模型')
    parse.add_argument('-embedding_size', dest='embedding_size', required=True, type=int, help='词向量纬度')
    parse.add_argument('-window_size', dest='window_size', required=True, type=int, help='上下文窗口大小')
    parse.add_argument('-epoch', dest='epoch', required=True, type=int, help='数据集重复次数')
    parse.add_argument('-batch', dest='batch', required=True, type=int, help='批处理的大小')
    parse.add_argument('-min_count', dest='min_count', required=True, type=int, help='词频最小要求值')
    parse.add_argument('-vocab_path', dest='vocab_path', help='已经存在的词典')

    args = parse.parse_args()

    updateArgs = Util.UpdateArgs()
    updateArgs.update(args)

    # 如果是指定了词汇表路径的话，那么直接从词汇表中构造vocab
    vocab = Vocab(args)
    if hasattr(args, 'vocab_path') and args.vocab_path is not None:
        vocab_path = os.path.join(args.out_folder, args.vocab_path)
        if os.path.exists(vocab_path):
            vocab.loadFromFile(vocab_path)
        else:
            raise FileNotFoundError()
    else:
        vocab.build()
        input_name = 'vocab' + Util.getFileName(args.input) + ".txt"
        vocab_path = os.path.join(args.out_folder, input_name)
        vocab.save(vocab_path)

    dataset = DataSet(args, vocab).getDataset().batch(args.batch)

    lr = 1e-3
    word2vec = Word2vec(args, vocab)
    word2vec.build(input_shape=(args.batch, 1))
    word2vec.summary()

    optimizer = tf.optimizers.Adam(lr=lr)

    for epoch in range(args.epoch):
        for step, (x, y) in enumerate(dataset):
            y = tf.reshape(y, shape=(args.batch, 1))
            with tf.GradientTape() as tape:
                embedd, nce_weights, nce_biases = word2vec(x)
                loss = tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    inputs=embedd,
                    labels=y,
                    num_sampled=64,
                    num_classes=len(vocab))
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, word2vec.trainable_weights)
            optimizer.apply_gradients(zip(grads, word2vec.trainable_weights))

            if step % 100 == 0:
                print(epoch, step, float(loss))


