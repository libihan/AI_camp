# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 20:28:49 2018

@author: Administrator
"""

from nltk.corpus import gutenberg
#有哪些语料在这个集合里？
print(gutenberg.fileids())

"""
words(x): 把x本书中的所有单词（包括标点符号）放到列表中中
"""
allwords = gutenberg.words('austen-emma.txt')
print(len(allwords))
print(len(set(allwords)))   #不重复的个数
allwords.count('Hamlet')

A = set(allwords)
longwords = [w for w in A if len(w)>12] #单词长度>12的所有单词
print(sorted(longwords))

from nltk.probability import FreqDist,ConditionalFreqDist
"""
FreqDist: 创建一个所给数据的频率分布
B(): 不同单词的个数
N(): 所有单词的个数
tabulate(20): 把前20组数据以表格的形式显示出来
fd2.plot(20,cumulative=True): 参数cumulative 对数据进行累计 
"""
fd2 = FreqDist([sx.lower() for sx in allwords if sx.isalpha()])
print("不同单词的个数：%d"%fd2.B())
print("所有单词的个数：%d"%fd2.N())
fd2.tabulate(20)    #把前20组数据 以表格的形式显示出来
fd2.plot(20)
fd2.plot(20,cumulative = True)

"""
freq('the')  #单词the出现的频率
ConditionalFreqDist( ): 条件频率统计的函数，研究类别之间的系统性的差异
"""
from nltk.corpus import inaugural
print(fd2.freq('the'))  #单词the出现的频率
cfd = ConditionalFreqDist(
        (fileid,len(w))
        for fileid in inaugural.fileids()
        for w in inaugural.word(fileid)
        if fileid > '1980' and fileid < '2010')
print(cfd.items())
cfd.plot()
