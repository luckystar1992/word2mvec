#coding:utf-8

"""
工具类，主要提供常用的函数使用
"""

import sys, os
import datetime, time


class OutputFolder:

    def __init__(self):
        pass

    def updateArgs(self, args):
        """更新参数"""
        # 每次词向量的保存都保存在out_folder文件夹中
        if not os.path.exists(args.out_folder):
            os.mkdir(args.out_folder)

        timestamp = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
        new_folder = '{out_folder}/training_{current}'.format(
            out_folder=args.out_folder,
            current=timestamp
        )
        os.mkdir(new_folder)

        # 向args中添加新的参数
        args.timestamp = timestamp
        args.out_folder = new_folder
