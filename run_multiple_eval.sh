#!/usr/bin/env bash
# 使用评测代码将ws353和scws两个数据集进行评测

echo '2020-0113 17:05'
for epoch in {9..9}
do
    python Evaluation.py -single 0 -day 0113 -time 1705 -epoch $epoch -threads 40
done

echo '2020-0114 07:42'
for epoch in {9..9}
do
    python Evaluation.py -single 1 -day 0114 -time 2357 -epoch $epoch -threads 40
done

echo '2020-0115 00:50'
for epoch in {9..9}
do
    python Evaluation.py -single 1 -day 0115 -time 0050 -epoch $epoch -threads 40
done

echo '2020-0115 01:42'
for epoch in {9..9}
do
    python Evaluation.py -single 1 -day 0115 -time 0142 -epoch $epoch -threads 40
done

echo '2020-0115 02:38'
for epoch in {9..9}
do
    python Evaluation.py -single 1 -day 0115 -time 0238 -epoch $epoch -threads 40
done

echo '2020 0115 03:32'
for epoch in {9..9}
do
    python Evaluation.py -single 0 -day 0115 -time 0332 -epoch $epoch -threads 40
done

echo '2020-0115 04:24'
for epoch in {9..9}
do
    python Evaluation.py -single 0 -day 0115 -time 0424 -epoch $epoch -threads 40
done

echo '2020-0115 05:29'
for epoch in {9..9}
do
    python Evaluation.py -single 0 -day 0115 -time 0529 -epoch $epoch -threads 40
done

echo '20200115 06:38'
for epoch in {9..9}
do
    python Evaluation.py -single 0 -day 0115 -time 0638 -epoch $epoch -threads 40
done

echo '20200115 07:42'
for epoch in {9..9}
do
    python Evaluation.py -single 0 -day 0115 -time 0742 -epoch $epoch -threads 40
done

echo '20200115 08:42'
for epoch in {9..9}
do
    python Evaluation.py -single 0 -day 0115 -time 0842 -epoch $epoch -threads 40
done

echo '20200115 09:44'
for epoch in {9..9}
do
    python Evaluation.py -single 0 -day 0115 -time 0944 -epoch $epoch -threads 40
done

echo '20200115 10:43'
for epoch in {9..9}
do
    python Evaluation.py -single 0 -day 0115 -time 1043 -epoch $epoch -threads 40
done

echo '20200115 12:01'
for epoch in {9..9}
do
    python Evaluation.py -single 0 -day 0115 -time 1201 -epoch $epoch -threads 40
done
