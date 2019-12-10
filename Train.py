#coding:utf-8

"""
训练过程
"""
import sys, argparse
import FileUtil
import Util
import Model
import math
import numpy as np
import time
import warnings
import tqdm
from multiprocessing import Value, Pool
from Huffman import Huffman

MAX_SEN_LEN = 1000              # 允许最长句子的单词数目

class UnigramTable:

    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.power = 0.75
        self.norm = sum([math.pow(vocabItem.count, self.power) for vocabItem in vocab])

        self.table_size = int(1e8)
        self.table = np.zeros(self.table_size, dtype=np.uint32)

    def build(self):
        print("Build a UnigramTable")
        p = 0
        i = 0
        for index, unigram in tqdm.tqdm(enumerate(self.vocab)):
            p += float(math.pow(unigram.count, self.power)) / self.norm
            while i < self.table_size and float(i) / self.table_size < p:
                self.table[i] = index
                i += 1

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1/(1 + math.exp(-z))

def init_process(*params):
    """线程的初始化，将子线程需要的参数全部传递过来"""
    global args, vocab, model, global_word_count, global_alpha, f_input
    args, vocab, model, global_word_count, global_alpha = params
    f_input = open(args.input, 'r')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        model.net1 = np.ctypeslib.as_array(model.net1)
        model.net2 = np.ctypeslib.as_array(model.net2)


def train_process(pid):
    """单个线程的训练"""
    epoch = args.epoch_index
    start = args.start_list[pid]
    end = args.end_list[pid]
    alpha_coeff = 1 - (global_word_count.value + epoch * vocab.word_count)/(vocab.word_count * args.epoch + 1)
    # 每个线程只处理corpus的某一部分
    f_input.seek(start)
    word_count = 0
    last_word_count = 0

    while f_input.tell() < end:
        # TODO: change readline to read one word each
        line = f_input.readline().strip()
        if not line:
            continue
        tokens = []
        tokens.append(FileUtil.BOL)
        tokens.extend(line.split())
        tokens.append(FileUtil.EOL)

        tokens = vocab.indices(tokens)

        # 对当前的句子进行遍历
        for word_index, token in enumerate(tokens):

            # 输出运行过程信息
            if global_word_count.value % int(vocab.word_count / 10000) == 0:
                # if pid == 0:
                #     sys.stdout.write("\r end-{end} current={current}".format(end=end, current=f_input.tell()))
                #     sys.stdout.flush()
                # 输出线程信息
                sys.stdout.write(
                    "\r𝑬-{epoch} 𝜃(⍺)={alpha_coeff:>4.2f} ⍺={alpha:>10.8f} ({current:>{len}d}/{total:>{len}d}){progress:>5.2f}٪".format(
                    epoch=epoch,
                    alpha_coeff=alpha_coeff,
                    alpha=global_alpha.value,
                    current=global_word_count.value,
                    len=len(str(vocab.word_count)),
                    total=vocab.word_count,
                    progress=float(global_word_count.value) / vocab.word_count * 100
                ))
                sys.stdout.flush()

            # 更新alpha
            if word_count - last_word_count > 10000:
                last_word_count = word_count
                # alpah 的衰减系数
                alpha_coeff = 1 - (global_word_count.value + epoch * vocab.word_count)/(vocab.word_count * args.epoch + 1)
                global_alpha.value = args.start_alpha * alpha_coeff
                if global_alpha.value < args.start_alpha * 0.0001:
                    global_alpha.value = args.start_alpha * 0.0001

            # 随机取上下文窗口大小
            current_window = np.random.randint(low=1, high=args.window_size + 1)
            context_start = max(word_index - current_window, 0)
            context_end = min(word_index + current_window + 1, len(tokens))
            context = tokens[context_start:word_index] + tokens[word_index + 1:context_end]

            if args.cbow:
                neu1 = np.mean(np.array([model.net1[c] for c in context]), axis=0)
                assert len(neu1) == args.embedding_size
                neu1e = np.zeros(args.embedding_size)
                if args.negative > 0:
                    classifiers = [(token, 1)] + [(target, 0) for target in args.table.sample(args.negative)]
                else:
                    classifiers = zip(vocab[token].path, vocab[token].code)

                for target, label in classifiers:
                    z = np.dot(neu1, model.net2[target])
                    p = sigmoid(z)
                    g = global_alpha.value * (label - p)
                    neu1e += g * model.net2[target]
                    model.net2[target] += g * neu1

                # 参数更新
                for context_word in context:
                    model.net1[context_word] += neu1e
            else:
                pass

            word_count += 1
            global_word_count.value += 1

    sys.stdout.write(
        "\r𝑬-{epoch} 𝜃(⍺)={alpha_coeff:>4.2f} ⍺={alpha:>10.8f} ({current:>{len}d}/{total:>{len}d}){progress:>5.2f}٪".format(
            epoch=epoch,
            alpha_coeff=alpha_coeff,
            alpha=global_alpha.value,
            current=global_word_count.value,
            len=len(str(vocab.word_count)),
            total=vocab.word_count,
            progress=float(global_word_count.value) / vocab.word_count * 100
        ))
    sys.stdout.flush()


def train(args):
    vocab = FileUtil.Vocab(args)
    vocab.build()

    singleModel = Model.SingleModel(args, vocab)
    singleModel.init_model()

    if args.negative > 0:
        print("Initializing Unigram Table")
        args.table = UnigramTable(vocab)
        args.table.build()
    else:
        print("Initializing Huffman Tree")
        huffman = Huffman(vocab)
        huffman.encode()
    vocab.save()

    # 开启多线程
    t0 = time.time()
    print("Begin Training with {0} threads.".format(args.num_threads))
    args.f_input = open(args.input)
    args.start_list, args.end_list = FileUtil.FileSplit().split(args, vocab)
    print(args.start_list)
    global_word_count = Value('i', 0)
    global_alpha = Value('f', args.alpha)
    for epoch in range(0, args.epoch):
        t_begin = time.time()
        global_word_count.value = 0
        args.epoch_index = epoch
        pool = Pool(processes=args.num_threads,
                    initializer=init_process,
                    initargs=(args, vocab, singleModel, global_word_count, global_alpha))
        pool.map(train_process, range(args.num_threads))
        t_end = time.time()
        print("\r𝑬-{epoch} ⍺={alpha:>10.8f} 𝑇={time:>10.2f}min  token/ps {speed:>6.1f}".format(
            epoch=epoch,
            alpha=global_alpha.value,
            time=(t_end - t_begin)/60,
            speed=vocab.word_count/(t_end-t_begin)/args.num_threads
        ), end='')
        singleModel.saveEmbedding(epoch)
    args.f_input.close()
    t1 = time.time()
    print("")
    print("Completed Training, Spend {spend_time:>10.2f} minutes.".format(spend_time=(t1-t0)/60))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', dest='input', required=True, help="训练语料的路径")
    parser.add_argument('-cbow', dest='cbow', required=True, help='使用的模型')
    parser.add_argument('-negative', dest='negative', type=int, required=True, help='求解模型的加速算法')
    parser.add_argument('-min_count', dest='min_count', type=int, help='词频最小要求值')
    parser.add_argument('-epoch', dest='epoch',  type=int, default=5, help='语料循环次数')
    parser.add_argument('-embedding_size', dest='embedding_size', required=True, type=int, help='词向量大小')
    parser.add_argument('-window_size', dest='window_size', required=True, type=int, help='上下文最大距离')
    parser.add_argument('-num_threads', dest='num_threads', default=2, type=int, help='开启的线程数')
    parser.add_argument('-binary', dest='binary', default=1, type=int, help='二进制保存词向量')
    parser.add_argument('-alpha', dest='alpha', default=0.025, type=float, help='初始alpha值')
    parser.add_argument('-out_folder', dest='out_folder', default='./out', help='模型/向量保存文件夹')

    args = parser.parse_args()
    args.start_alpha = args.alpha
    outputFolder = Util.OutputFolder()
    outputFolder.updateArgs(args)
    # 正式训练
    train(args)

