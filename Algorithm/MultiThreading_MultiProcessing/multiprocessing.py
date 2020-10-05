#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    multiprocessing.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        4/16/20 11:19 PM
# @torch: tensor.method(in-place) or torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from time import ctime,sleep
import multiprocessing

# 定义两个方法说和写
def talk(content,loop):
    for i in range(loop):
        print("Talk:%s %s" %(content,ctime()))
        sleep(2)

def write(content,loop):
    for i in range(loop):
        print("Write:%s %s" %(content,ctime()))
        sleep(3)

#定义两个进程
process = []
p1 = multiprocessing.Process(target=talk,args=('hello,51zxw',2))
process.append(p1)

p2 = multiprocessing.Process(target=write,args=('Python',2))
process.append(p2)
#调用进程
if __name__=='__main__':
    core = multiprocessing.cpu_count()
    print(core)
    for p in process:
        p.start()
    for p in process:
        p.join()
    print("All process is Run! %s" %ctime())
