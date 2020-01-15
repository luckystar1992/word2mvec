#!/usr/bin/env bash
#使用原始word2vec的训练代码进行训练

CORPUS="data/corpus/wiki_text8"
echo 'Begin Training'

python Train.py -input ${CORPUS} \
       -cbow 1 \
       -negative 0 \
       -min_count 5 \
       -epoch 10 \
       -embedding_size 100 \
       -window_size 5 \
       -num_threads 24 \
       -binary 0 \
       -senses 3