#coding:utf-8

import sys, os
import Util
import tensorflow as tf
import numpy as np
import tqdm

"""
文件处理成Dictionary的工具类
"""

BOL = '<BOL>'
EOL = '<EOL>'
UNK = '<UNK>'

class VocabItem:

    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None
        self.code = None


class Vocab:

    def __init__(self, args):
        self.f_input = open(args.input, 'r')
        self.args = args
        self.word_count = 0
        self.vocab_items = list()
        self.vocab_hash = dict()

        vocab_path = os.path.join(self.args.out_folder, 'vocab' + self.args.input + '.txt')

    def __len__(self):
        return len(self.vocab_items)

    def __getitem__(self, i):
        return self.vocab_items[i]

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash

    def loadFromFile(self, vocab_path):
        """如果指定了词典存储位置，则直接从词典加载Vocab"""
        print("Load vocab from ", vocab_path)
        vocab_items = list()
        vocab_hash = dict()
        with open(vocab_path) as f:
            for line in f.readlines():
                line = line.strip().split(" ", 2)
                count = int(line[0])
                word = line[1]
                vocab = VocabItem(word)
                vocab.count = count
                vocab_hash[word] = len(vocab_items)
                vocab_items.append(vocab)

        self.vocab_items = vocab_items
        self.vocab_hash = vocab_hash
        self.bytes = Util.getFileBytesCount(self.args.input)
        self.word_count = Util.getFileWordCount(self.args.input)



    def __sort__(self):
        """将少于min_count个出现次数的词汇转成UNK并反序排序"""
        tmp = list()
        tmp.append(VocabItem(UNK))
        unk_hash = 0
        count_unk = 0

        for vocab in self.vocab_items:
            if vocab.count < self.args.min_count:
                count_unk += 1
                tmp[unk_hash].count += vocab.count
            else:
                tmp.append(vocab)
        tmp.sort(key=lambda vocab: vocab.count, reverse=True)

        # 更新vocab_hash
        vocab_hash = dict()
        for index, vocab in enumerate(tmp):
            vocab_hash[vocab.word] = index

        self.vocab_items = tmp
        self.vocab_hash = vocab_hash
        print("Unknown Vocab Size: {unk_num}".format(unk_num=count_unk))

    def build(self):
        """构造词典"""
        print("Build vocab by iterate ", self.args.input)
        self.vocab_items.append(VocabItem(BOL))
        self.vocab_items.append(VocabItem(EOL))
        self.vocab_hash[BOL] = 0
        self.vocab_hash[EOL] = 1

        # 第一次遍历从文件中读取的数据集
        for line in tqdm.tqdm(self.f_input):
            line = line.strip()
            tokens = line.split()
            for token in tokens:
                if token not in self.vocab_hash:
                    self.vocab_hash[token] = len(self.vocab_items)
                    self.vocab_items.append(VocabItem(token))
                self.vocab_items[self.vocab_hash[token]].count += 1
                self.word_count += 1

                # if self.word_count % 1000 == 0:
                #     sys.stdout.write("\rReading Words {word_count:>10.2f}k".format(word_count=self.word_count/1000.0))
                #     sys.stdout.flush()

            self.vocab_items[self.vocab_hash[BOL]].count += 1
            self.vocab_items[self.vocab_hash[EOL]].count += 1
            self.word_count += 2

        # 第二次遍历内存中的数据集
        print("")
        self.__sort__()

        self.bytes = self.f_input.tell()
        print("Total Words: {word_cout}".format(word_cout=self.word_count))
        print("Total Bytes: {bytes}".format(bytes=self.bytes))
        print("Vocab Size : {vocab_size}".format(vocab_size=len(self)))

    def save(self, vocab_path):
        """将Vocab信息输出到out_folder中"""
        with open(vocab_path, 'w') as f:
            for vocab in self:
                f.write("%6d %s\n" % (vocab.count, vocab.word))

    def getMostVocab(self):
        """将词汇表限制在一定长度以内"""
        if self.args.vocab_size < len(self):
            self.vocab_items = self.vocab_items[0:self.args.vocab_size]
            self.vocab_hash = {vocab.word: index for index, vocab in enumerate(self.vocab_items)}
            print("Vocab Size (after truncated): {vocab_size}".format(vocab_size=len(self)))

    def indices(self, tokens):
        """将一句话转成相应的index表示"""
        return [self.vocab_hash[token] if token in self else self.vocab_hash[UNK] for token in tokens]


class FileSplit:
    """将整个的文件分为python3可以按照utf-8正常解码的各个部分"""
    splitIndexList = list()

    @classmethod
    def split(cls, args, vocab):
        """分文件"""
        index_list = [int(vocab.bytes / args.num_threads * pid) for pid in range(1, args.num_threads + 1)]
        new_index_list = []
        f_input = args.f_input
        for index in index_list:
            f_input.seek(index)
            while True:
                try:
                    f_input.read(1)
                    break
                except:
                    index += 1
                    f_input.seek(index)
            new_index_list.append(index)

        return [0] + new_index_list[:-1], new_index_list


class DataSet:
    """使用TF的data.Dataset方式来返回"""

    def __init__(self, args, vocab, once=False):
        """构造函数
        :param args:    参数对象
        :param vocab:   词汇表统计对象
        :param once:    是否一次加载到内存中
        """
        self.args = args
        self.vocab = vocab
        self.once = once

    def getDataOnce(self):
        """一次性将所有数据加载到内存中"""
        print("Once")
        train_x, train_y = list(), list()
        with open(self.args.input) as f_input:
            for line in tqdm.tqdm(f_input.readlines()):
                tokens = line.strip().split()
                tokens_indices = self.vocab.indices(tokens)
                for index, target_word in enumerate(tokens_indices):
                    context_words = list()
                    begin = index - self.args.window_size if index - self.args.window_size > 0 else 0
                    end = index + 1 + self.args.window_size if index + self.args.window_size + 1 < len(tokens_indices) else len(
                        tokens_indices)
                    context_words.extend(tokens_indices[begin:index])
                    context_words.extend(tokens_indices[index + 1:end])
                    if self.args.cbow > 0:
                        train_x.append(context_words)
                        train_y.append(target_word)
                    else:
                        for i in range(len(context_words)):
                            train_x.append(target_word)
                            train_y.append(context_words[i])

        return np.array(train_x), np.array(train_y)

    def generator(self):
        """读取文本文件，并生成CBOW或是Skip-Gram的sample"""
        with open(self.args.input) as f_input:
            for line in f_input.readlines():
                tokens = line.strip().split()
                tokens_indices = self.vocab.indices(tokens)
                for index, target_word in enumerate(tokens_indices):
                    context_words = list()
                    begin = index - self.args.window_size if index - self.args.window_size > 0 else 0
                    end = index + 1 + self.args.window_size if index + self.args.window_size + 1 < len(tokens_indices) else len(
                        tokens_indices)
                    context_words.extend(tokens_indices[begin:index])
                    context_words.extend(tokens_indices[index + 1:end])
                    if self.args.cbow > 0:
                        yield context_words, target_word
                    else:
                        for i in range(len(context_words)):
                            yield target_word, context_words[i]

    def getDataset(self):
        """获取数据集， 根据是一次性加载到内存中，还是使用generator进行逐步读取"""
        if self.once:
            np_train_x, np_train_y = self.getDataOnce()
            print(np_train_x.shape)
            print(np_train_y.shape)
            dataset = tf.data.Dataset.from_tensor_slices((np_train_x, np_train_y))
        else:
            if self.args.cbow:
                dataset = tf.data.Dataset.from_generator(
                    self.generator, (tf.int32, tf.int32), ((None,), ())
                )
            else:
                dataset = tf.data.Dataset.from_generator(
                    self.generator, (tf.int32, tf.int32), ((), ())
                )

        return dataset


if __name__ == '__main__':
    import argparse
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

    updateArgs = Util.ArgsConfig()
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

    dataset = DataSet(args, vocab).getDataset().batch(8)
    for step, (x, y) in enumerate(dataset):
        if step % 1000 == 0:
            print(step, x.shape, y.shape)


