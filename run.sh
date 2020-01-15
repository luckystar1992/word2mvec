#!/usr/bin/env bash

CORPUS="data/corpus/wiki_text8"
echo 'Begin Training'
# 单一语境的训练过程
python  Train.py -input ${CORPUS} \
        -cbow 1 \
        -negative 0 \
        -min_count 5 \
        -epoch 10 \
        -embedding_size 50 \
        -window_size 5 \
        -num_threads 40 \
        -binary 0 \
        -alpha 0.05 \
        -senses -1

python  Train.py -input ${CORPUS} \
        -cbow 1 \
        -negative 0 \
        -min_count 5 \
        -epoch 10 \
        -embedding_size 100 \
        -window_size 5 \
        -num_threads 40 \
        -binary 0 \
        -alpha 0.05 \
        -senses -1

python  Train.py -input ${CORPUS} \
        -cbow 1 \
        -negative 0 \
        -min_count 5 \
        -epoch 10 \
        -embedding_size 200 \
        -window_size 5 \
        -num_threads 40 \
        -binary 0 \
        -alpha 0.05 \
        -senses -1

python  Train.py -input ${CORPUS} \
        -cbow 1 \
        -negative 0 \
        -min_count 5 \
        -epoch 10 \
        -embedding_size 300 \
        -window_size 5 \
        -num_threads 40 \
        -binary 0 \
        -alpha 0.05 \
        -senses -1

# 多语境的训练过程
python  Train.py -input ${CORPUS} \
        -cbow 1 \
        -negative 0 \
        -min_count 5 \
        -epoch 10 \
        -embedding_size 100 \
        -window_size 5 \
        -num_threads 40 \
        -binary 0 \
        -alpha 0.05 \
        -senses 0

python  Train.py -input ${CORPUS} \
        -cbow 1 \
        -negative 0 \
        -min_count 5 \
        -epoch 10 \
        -embedding_size 100 \
        -window_size 5 \
        -num_threads 40 \
        -binary 0 \
        -alpha 0.05 \
        -senses 5 \
        -sim_threshold 0.6

python  Train.py -input ${CORPUS} \
        -cbow 1 \
        -negative 0 \
        -min_count 5 \
        -epoch 10 \
        -embedding_size 100 \
        -window_size 5 \
        -num_threads 40 \
        -binary 0 \
        -alpha 0.05 \
        -senses 5 \
        -sim_threshold 0.62

python  Train.py -input ${CORPUS} \
        -cbow 1 \
        -negative 0 \
        -min_count 5 \
        -epoch 10 \
        -embedding_size 100 \
        -window_size 5 \
        -num_threads 40 \
        -binary 0 \
        -alpha 0.05 \
        -senses 5 \
        -sim_threshold 0.64

python  Train.py -input ${CORPUS} \
        -cbow 1 \
        -negative 0 \
        -min_count 5 \
        -epoch 10 \
        -embedding_size 100 \
        -window_size 5 \
        -num_threads 40 \
        -binary 0 \
        -alpha 0.05 \
        -senses 5 \
        -sim_threshold 0.66

python  Train.py -input ${CORPUS} \
        -cbow 1 \
        -negative 0 \
        -min_count 5 \
        -epoch 10 \
        -embedding_size 100 \
        -window_size 5 \
        -num_threads 40 \
        -binary 0 \
        -alpha 0.05 \
        -senses 5 \
        -sim_threshold 0.68

python  Train.py -input ${CORPUS} \
        -cbow 1 \
        -negative 0 \
        -min_count 5 \
        -epoch 10 \
        -embedding_size 100 \
        -window_size 5 \
        -num_threads 40 \
        -binary 0 \
        -alpha 0.05 \
        -senses 5 \
        -sim_threshold 0.70

python  Train.py -input ${CORPUS} \
        -cbow 1 \
        -negative 0 \
        -min_count 5 \
        -epoch 10 \
        -embedding_size 100 \
        -window_size 5 \
        -num_threads 40 \
        -binary 0 \
        -alpha 0.05 \
        -senses 10 \
        -sim_threshold 0.65

python  Train.py -input ${CORPUS} \
        -cbow 1 \
        -negative 0 \
        -min_count 5 \
        -epoch 10 \
        -embedding_size 100 \
        -window_size 5 \
        -num_threads 40 \
        -binary 0 \
        -alpha 0.05 \
        -senses 10 \
        -sim_threshold 0.70
