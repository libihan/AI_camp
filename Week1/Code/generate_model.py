# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:17:28 2018

@author: Administrator
"""

# 用gensim打开glove词向量需要在向量的开头增加一行：所有的单词数 词向量的维度  
#计算行数，就是单词数  
def getFileLineNums(filename):  
    f = open(filename, 'r')  
    count = 0  
    for line in f:  
        count += 1  
    return count  
  
def prepend_slow(infile, outfile, line):  
    with open(infile, 'r') as fin:  
        with open(outfile, 'w') as fout:  
            fout.write(line + "\n")  
            for line in fin:  
                fout.write(line)  
  
def load(filename):
    num_lines = getFileLineNums(filename)
    gensim_file = 'word2vec_model.txt'  
    gensim_first_line = "{} {}".format(num_lines, 300)  
    # Prepends the line.  
    prepend_slow(filename, gensim_file, gensim_first_line)  
      
if __name__ == '__main__':
    load(r'C:\Users\Administrator\Desktop\AI_Camp\simple.word2vec.300d.txt')  