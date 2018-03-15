# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:33:01 2018

@author: Administrator
"""
import re
import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV,train_test_split
from scipy.stats import pearsonr 


#==============================================================================
#  数据预处理
#==============================================================================
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def preProcess(sentence):
    # 去除标点符号
    pattern1 = '[,.:;?()!@#%$*=]'
    sentence = re.sub(pattern1,'',sentence)
  
    # 分词 ---> 转成小写 ---> 去除停用词 ---> 删除非字母数字字符
    stopword = stopwords.words('english')
    sentence = [word for word in word_tokenize(sentence.lower()) if word not in stopword and word.isalnum() and len(word)>=3]
    
    # 单词变体还原
    lemmatizer = WordNetLemmatizer()
    sentence = [lemmatizer.lemmatize(word) for word in sentence]
    
    return sentence

#==============================================================================
#  特征提取
#==============================================================================
#from gensim import corpora, models,similarities
#
#def getFeature(text1,text2):
#    dictionary = corpora.Dictionary(text1+text2)    #为每个出现在语料库中的单词分配了一个独一无二的整数编号
#    #print(dictionary.token2id) #查看单词与编号之间的映射关系  
#    corpus1 = [dictionary.doc2bow(text) for text in text1] #函数doc2bow()简单地对每个不同单词的出现次数进行了计数，并将单词转换为其编号，然后以稀疏向量的形式返回结果 
#    tfidf = models.TfidfModel(corpus1)  #将bow向量中的特征进行IDF统计
#    corpus1_tfidf = tfidf[corpus1]
#
#    corpus2 = [dictionary.doc2bow(text) for text in text2]
#    corpus2_tfidf = tfidf[corpus2]
#    
#    index = similarities.MatrixSimilarity(corpus1_tfidf) 
#    sims = index[corpus2_tfidf]
#    tfidf_features = np.mat(np.diag(sims)).T
#    
#    # LSI模型
#    lsi = models.LsiModel(corpus1_tfidf, id2word=dictionary, num_topics=len(text1))
#    index = similarities.MatrixSimilarity(lsi[corpus1])
#    sims = index[lsi[corpus2]]
#    lsi_features = np.mat(np.diag(sims)).T
#    
#    return tfidf_features,lsi_features

if __name__ == '__main__':
    #==============================================================================
    # 读取数据
    #==============================================================================
    train_data = pd.read_csv(r'C:\Users\Administrator\Desktop\AI_Camp\train_data.csv')
    test_features = pd.read_csv(r'C:\Users\Administrator\Desktop\AI_Camp\test_features.csv')
    #==============================================================================
    # 载入train和test数据集
    #==============================================================================
    dataSet = train_data.iloc[:,:-1]
    labelSet = train_data['score']
    
    train_text, test_text, train_labels, test_labels = train_test_split(dataSet, labelSet, test_size=0.33,random_state=23333)
    
#    train_text1 = [preProcess(x) for x in list(train_text[1])]
#    train_text2 = [preProcess(x) for x in list(train_text[2])]
#    
#    test_text1 = [preProcess(x) for x in list(test_text[1])]
#    test_text2 = [preProcess(x) for x in list(test_text[2])]
    #==============================================================================
    # 特征提取
    #==============================================================================
#    train_features = np.hstack(getFeature(train_text1,train_text2)) #在行上合并
    #==============================================================================
    # 模型训练
    #==============================================================================
    parameters = {'kernel':['rbf'], 'C':[10,100,500,1e3,2e3,3e3], 'gamma': [1e-6,1e-5,1e-4,1e-3,1e-2,0.1, 1, 10]}
    svr = SVR()
    clf = GridSearchCV(svr, parameters, n_jobs=-1)
#    clf = SVR(kernel='rbf', C=20, gamma=1)
    clf.fit(train_text,train_labels)
#    
#    print('The parameters of the best model are: ')
#    print(clf.best_params_)
    
    preds = clf.predict(train_text)
    print('The pearsonr of training set is {}'.format(pearsonr (list(train_labels), list(preds))[0]))
    print('The MSE of training set is {}'.format(mean_squared_error(list(train_labels), list(preds))))
      
    #==============================================================================
    # 预测 测试集
    #==============================================================================   
#    test_features = np.hstack(getFeature(test_text1,test_text2))
    preds = clf.predict(test_text)
    
    print('The pearsonr of test set is {}'.format(pearsonr (list(test_labels), list(preds))[0]))
    print('The MSE of test set is {}'.format(mean_squared_error(list(test_labels), list(preds))))