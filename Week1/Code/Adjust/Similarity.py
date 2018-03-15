# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:33:01 2018

@author: Administrator
"""
import pandas as pd
import numpy as np
from gensim.models import word2vec
import re
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

def preProcess(text):
    pattern1 = '[,.]'
    pattern2 ="(it's)|(It's)"
    
    text = re.sub(pattern1,'',text)
    text = re.sub(pattern2,'it is',text)
    text = text.lower()
    return text


data_path = r'C:\Users\Administrator\Desktop\AI_Camp\Tal_SenEvl_train_136KB.txt'
data = pd.read_table(data_path,header = None)
data.head()


#==============================================================================
# ## 载入train和test数据集
#==============================================================================
dataSet = data.loc[:,1:2]
labelSet = data.loc[:,3]

train_text, test_text, train_labels, test_labels = train_test_split(dataSet, labelSet, test_size=0.33, random_state=233230)

train_text1 = [preProcess(x) for x in list(train_text[1])]
train_text2 = [preProcess(x) for x in list(train_text[2])]
train_label = list(train_labels)

#==============================================================================
# 数据分析
#==============================================================================
# 分数分布
#plt.figure(figsize=(20,10)) 
#train_labels.value_counts().plot(kind="bar")
#plt.show()
#
#
## 句子字符数量分布
#train_ts = pd.Series(train_text[1].tolist() + train_text[2].tolist()).astype(str)
#test_ts = pd.Series(test_text[1].tolist() + test_text[2].tolist()).astype(str)
#
#dist_train = train_ts.apply(len)
#dist_test = test_ts.apply(len)
#
#plt.figure(figsize=(15, 10))
#pal = sns.color_palette()
#plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], normed=True, label='train')
#plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='test')
#plt.title('Normalised histogram of character count in questions', fontsize=15)
#
#plt.xlabel('Number of characters', fontsize=15)
#plt.ylabel('Probability', fontsize=15)
#plt.show()
#print('mean-train {:.2f} std-train {:.2f}, max-train {:.2f}'.format(dist_train.mean(), 
#                          dist_train.std(),dist_train.max()))
#
## 句子词数目分布
#dist_train = train_ts.apply(lambda x: len(x.split()))
#dist_test = test_ts.apply(lambda x: len(x.split()))
#plt.figure(figsize=(15, 10))
#plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], normed=True, label='train')
#plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='test')
#plt.title('Normalised histogram of word count in questions', fontsize=15)
#plt.legend()
#plt.xlabel('Number of words', fontsize=15)
#plt.ylabel('Probability', fontsize=15)
#plt.show()
#print('mean-train {:.2f} std-train {:.2f}, max-train {:.2f}'.format(dist_train.mean(), 
#                          dist_train.std(),dist_train.max()))

#==============================================================================
# 模型训练
#==============================================================================
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr 

train_set = train_text1 + train_text2
tf_vector = CountVectorizer()
bow_features = tf_vector.fit_transform(train_set)

feature_names = tf_vector.get_feature_names()
print("total vocab num is {}".format(len(feature_names)))

text1_features = tf_vector.transform(train_text1)
text2_features = tf_vector.transform(train_text2)
train_features = text1_features + text2_features    #词向量简单相加
print(train_features.shape)

#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2, 5, 10, 20, 100], 'gamma': [0.01, 0.1, 1, 10]}
parameters = {'kernel':['rbf'], 'C':[1000], 'gamma': [0.01,0.1]}
svr = SVR()
clf = GridSearchCV(svr, parameters, n_jobs=-1)
clf.fit(train_features,train_label)

print('The parameters of the best model are: ')
print(clf.best_params_)

preds = clf.predict(train_features)
pearsonr (train_label, preds)

test_text1 = [preProcess(x) for x in list(test_text[1])]
test_text2 = [preProcess(x) for x in list(test_text[2])]
test_label = list(test_labels)

test_features = tf_vector.transform(test_text1) + tf_vector.transform(test_text2)
preds = clf.predict(test_features)
pearsonr (test_label, preds)