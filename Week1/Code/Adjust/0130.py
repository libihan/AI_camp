# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:14:54 2018

@author: Administrator
"""

texts = [['human', 'interface', 'computer'],
 ['survey', 'user', 'computer', 'system', 'response', 'time'],
 ['eps', 'user', 'interface', 'system'],
 ['system', 'human', 'system', 'eps'],
 ['user', 'response', 'time'],
 ['trees'],
 ['graph', 'trees'],
 ['graph', 'minors', 'trees'],
 ['graph', 'minors', 'survey']]

#==============================================================================
# Step 1. 训练语料的预处理
#==============================================================================
from gensim import corpora
"""
得到了语料中每一篇文档对应的稀疏向量（这里是bow向量）
向量的每一个元素代表了一个word在这篇文档中出现的次数
"""
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus[0]) # [(0, 1), (1, 1), (2, 1)]

#==============================================================================
# Step 2. 主题向量的变换
#==============================================================================
from gensim import models
tfidf = models.TfidfModel(corpus)

doc_bow = [(0, 1), (1, 1)]
print(tfidf[doc_bow])

#==============================================================================
# Step 3. 文档相似度的计算
#==============================================================================
"""
# 构造LSI模型并将待检索的query和文本转化为LSI主题向量
# 转换之前的corpus和query均是BOW向量
"""
lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
documents = lsi_model[corpus]
query_vec = lsi_model[doc_bow]

index = similarities.MatrixSimilarity(documents)



