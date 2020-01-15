#!/usr/bin/env bash
#使用原始word2vec的训练代码进行训练

CORPUS="data/corpus/wiki_text8"
echo 'Begin Training'

mkdir -p out/word2vec
./bin/word2vec -train ${CORPUS} -output out/word2vec/text8_cbow_hs_100.txt \
    -size 100 \
    -window 5 \
    -cbow 1 \
    -hs 1 \
    -negative 0 \
    -threads 24 \
    -iter 10 \
    -min-count 5 \
    -binary 0 \
    -save-vacab out/word2vec/text8.vocab