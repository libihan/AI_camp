# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:33:01 2018

@author: Administrator
"""
import re
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
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
from gensim import corpora, models,similarities

def getFeature(text1,text2):
    dictionary = corpora.Dictionary(text1+text2)    #为每个出现在语料库中的单词分配了一个独一无二的整数编号
    #print(dictionary.token2id) #查看单词与编号之间的映射关系  
    corpus1 = [dictionary.doc2bow(text) for text in text1] #函数doc2bow()简单地对每个不同单词的出现次数进行了计数，并将单词转换为其编号，然后以稀疏向量的形式返回结果 
    tfidf = models.TfidfModel(corpus1)  #将bow向量中的特征进行IDF统计
    corpus1_tfidf = tfidf[corpus1]

    corpus2 = [dictionary.doc2bow(text) for text in text2]
#    corpus2_tfidf = tfidf[corpus2]
#    
#    index = similarities.MatrixSimilarity(corpus1_tfidf) 
#    sims = index[corpus2_tfidf]
#    tfidf_features = np.mat(np.diag(sims)).T
    
    # LSI模型
    lda = models.LdaModel(corpus1_tfidf, id2word=dictionary, num_topics=len(text1)/100)
    index = similarities.MatrixSimilarity(lda[corpus1])
    sims = index[lda[corpus2]]
    lda_features = np.mat(np.diag(sims)).T
    
    return lda_features

if __name__ == '__main__':
    #==============================================================================
    # 读取数据
    #==============================================================================
    data_path = r'C:\Users\Administrator\Desktop\AI_Camp\Tal_SenEvl_train_136KB.txt'
    data = pd.read_table(data_path,header = None)
    data.head()
    
    #==============================================================================
    # 载入train和test数据集
    #==============================================================================
    dataSet = data.loc[:,1:2]
    labelSet = data.loc[:,3]
    
    train_text, test_text, train_labels, test_labels = train_test_split(dataSet, labelSet, test_size=0.33)
    
    train_text1 = [preProcess(x) for x in list(train_text[1])]
    train_text2 = [preProcess(x) for x in list(train_text[2])]
    
    test_text1 = [preProcess(x) for x in list(test_text[1])]
    test_text2 = [preProcess(x) for x in list(test_text[2])]
    #==============================================================================
    # 特征提取
    #==============================================================================
    train_features = getFeature(train_text1,train_text2)
    
    #==============================================================================
    # 模型训练
    #==============================================================================
    parameters = {'kernel':['rbf'], 'C':[1,10,20,100,200,500], 'gamma': [0.01, 0.1, 1, 10,100,200]}
    #parameters = {'kernel':['rbf'], 'C':[1000], 'gamma': [0.1]}
    svr = SVR()
    clf = GridSearchCV(svr, parameters, n_jobs=-1)
    clf.fit(train_features,train_labels)
    
    print('The parameters of the best model are: ')
    print(clf.best_params_)
    
    preds = clf.predict(train_features)
    print('The pearsonr of training set is {}'.format(pearsonr (list(train_labels), list(preds))[0]))
    print('The MSE of training set is {}'.format(mean_squared_error(list(train_labels), list(preds))))
      
    #==============================================================================
    # 预测 测试集
    #==============================================================================   
    test_features = getFeature(test_text1,test_text2)
        
    preds = clf.predict(test_features)
    
    print('The pearsonr of test set is {}'.format(pearsonr (list(test_labels), list(preds))[0]))
    print('The MSE of test set is {}'.format(mean_squared_error(list(test_labels), list(preds))))