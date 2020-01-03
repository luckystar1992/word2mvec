#!/usr/bin/env bash
#将python实现的word2vec进行不同参数配置下的实验比较

# CBOW + Huffman
python Train.py -input ctrip -cbow 1 -vocab_path vocabctrip.txt -negative 0 -min_count 5 -epoch 10 \
     -embedding_size 100 -window_size 5 -num_threads 24 -binary 0 -out_folder out

# CBOW + Negative Sampling
python Train.py -input ctrip -cbow 1 -vocab_path vocabctrip.txt -negative 10 -min_count 5 -epoch 10 \
     -embedding_size 100 -window_size 5 -num_threads 24 -binary 0 -out_folder out

# Skip-Gram + Huffman
python Train.py -input ctrip -cbow 0 -vocab_path vocabctrip.txt -negative 0 -min_count 5 -epoch 10 \
     -embedding_size 100 -window_size 5 -num_threads 24 -binary 0 -out_folder out

# Skip-Gram + Negative Sampling
python Train.py -input ctrip -cbow 0 -vocab_path vocabctrip.txt -negative 10 -min_count 5 -epoch 10 \
     -embedding_size 100 -window_size 5 -num_threads 24 -binary 0 -out_folder out
