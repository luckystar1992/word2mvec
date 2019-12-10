# word2mvec
word embedding - python version
----------------

The python code of word2vec was base on deborausujono's work [code](https://github.com/deborausujono/word2vecpy/blob/master/word2vec.py)
and Google's work [code](http://code.google.com/p/word2vec/)

## Usage
----------------

```bash
python Train.py -input text.txt -cbow 1 -negative 0 -min_count 5 -epoch 10 -embedding_size 100
-num_threads 8 -binary 0 -alpha 0.025 -out_folder out
```