#coding:utf-8

import sys, os

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

    def __len__(self):
        return len(self.vocab_items)

    def __getitem__(self, i):
        return self.vocab_items[i]

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash

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
        self.vocab_items.append(VocabItem(BOL))
        self.vocab_items.append(VocabItem(EOL))
        self.vocab_hash[BOL] = 0
        self.vocab_hash[EOL] = 1

        # 第一次遍历从文件中读取的数据集
        for line in self.f_input:
            line = line.strip()
            tokens = line.split()
            for token in tokens:
                if token not in self.vocab_hash:
                    self.vocab_hash[token] = len(self.vocab_items)
                    self.vocab_items.append(VocabItem(token))
                self.vocab_items[self.vocab_hash[token]].count += 1
                self.word_count += 1

                if self.word_count % 1000 == 0:
                    sys.stdout.write("\rReading Words {word_count:>10.2f}k".format(word_count=self.word_count/1000.0))
                    sys.stdout.flush()

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

    def save(self):
        """将Vocab信息输出到文件中"""
        vocab_path = os.path.join(self.args.out_folder, 'vocab.txt')
        with open(vocab_path, 'w') as f:
            for vocab in self:
                f.write("%6d %s\n" % (vocab.count, vocab.word))


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
