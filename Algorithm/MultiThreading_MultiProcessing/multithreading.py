#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence 
# @Filename:    multithreading.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        4/16/20 11:16 PM
# @torch: tensor.method(in-place) or torch.method(tensor)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# from time import sleep,ctime
# import threading
#
#
# # 定义说和写的方法
# def talk(content,loop):
#     for i in range(loop):
#         print("Start talk:%s %s" % (content,ctime()))
#         sleep(2)
#
#
# def write(content, loop):
#      for i in range(loop):
#          print("Start write:%s %s" % (content,ctime()))
#          sleep(3)
#
#
# # 定义和加载说和写的线程
# threads = []
# threads.append(threading.Thread(target=talk,args=('Hello 51zxw',2)))
# # threads.append(t1)
# threads.append(threading.Thread(target=write,args=('Life is short,You need Python',2)))
# # threads.append(t2)
#
# # 执行多线程
# if __name__=='__main__':
#     for t in threads:
#         t.start()
#     for t in threads:
#         t.join()
#     print("All Thread end! %s" %ctime())



# -*- encoding: gb2312 -*-
# import string, threading, time
#
#
# def thread_main(a):
#     global count, mutex
#     # 获得线程名
#     threadname = threading.currentThread().getName()
#
#     for x in range(0, int(a)):
#         # 取得锁
#         mutex.acquire()
#         count = count+1
#         # 释放锁
#         mutex.release()
#         print(threadname, x, count)
#         time.sleep(1)
#
#
# def main(num):
#     global count, mutex
#     threads = []
#
#     count = 1
#     # 创建一个锁
#     mutex = threading.Lock()
#     # 先创建线程对象
#     for x in range(0, num):
#         threads.append(threading.Thread(target=thread_main, args=(10,)))
#     # 启动所有线程
#     for t in threads:
#         t.start()
#     # 主线程中等待所有子线程退出
#     for t in threads:
#         t.join()
#
#
# if __name__ == '__main__':
#     num = 4
#     # 创建4个线程
#     main(4)



import sys
import threading
import queue
q = queue.Queue()


def worker1(x, y):
	func_name = sys._getframe().f_code.co_name
	print("%s run ..." % func_name)
	q.put((x + y, func_name))


def worker2(x, y):
	func_name = sys._getframe().f_code.co_name
	print("%s run ...." % func_name)
	q.put((x - y, func_name))


if __name__ == '__main__':
	result = list()
	t1 = threading.Thread(target=worker1, name='thread1', args=(10, 5, ))
	threads = []
	s = [i for i in range(3)]
	for i in range(3):
		t2 = threading.Thread(target=worker2, name='thread2', args=(20, s[i], ))
		threads.append(t2)
	print('-' * 50)
	t1.start()
	for t2 in threads:
		t2.start()
	t1.join()
	for t2 in threads:
		t2.join()
	while not q.empty():
		result.append(q.get())
	print(result)
	for item in result:
		if item[1] == worker1.__name__:
			print("%s 's return value is : %s" % (item[1], item[0]))
		elif item[1] == worker2.__name__:
			print("%s 's return value is : %s" % (item[1], item[0]))