#coding:utf-8

"""
使用数组的方式或是使用二叉树的结构进行huffman树的建立
"""


class Node:

    def __init__(self, count):
        self.count = count
        self.left = None
        self.right = None
        self.parent = None


class Huffman:

    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(self.vocab.vocab_items)

    def encode(self):
        count = [vocab.count for vocab in self.vocab.vocab_items] + [1e15] * (self.vocab_size - 1)
        parent = [0] * (2 * self.vocab_size - 2)
        binary = [0] * (2 * self.vocab_size - 2)

        pos1 = self.vocab_size - 1
        pos2 = self.vocab_size

        for index in range(self.vocab_size-1):
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1 = pos1
                    pos1 -= 1
                else:
                    min1 = pos2
                    pos2 += 1
            else:
                min1 = pos2
                pos2 += 1

            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2 = pos1
                    pos1 -= 1
                else:
                    min2 = pos2
                    pos2 += 1
            else:
                min2 = pos2
                pos2 += 1

            count[self.vocab_size + index] = count[min1] + count[min2]
            parent[min1] = self.vocab_size + index
            parent[min2] = self.vocab_size + index
            binary[min2] = 1

        root_index = 2 * self.vocab_size - 2
        for index, token in enumerate(self.vocab.vocab_items):
            path = []
            code = []

            node_index = index
            while node_index < root_index:
                if node_index >= self.vocab_size: path.append(node_index)
                code.append(binary[node_index])
                node_index = parent[node_index]
            path.append(root_index)

            token.path = [i - self.vocab_size for i in path[::-1]]
            token.code = code[::-1]
