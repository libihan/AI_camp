# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:33:01 2018

@author: Administrator
"""
import re
import pandas as pd
import numpy as np


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr 
import matplotlib.pyplot as plt


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
    
    V = []
    for i in range(0,12):
        train_text, test_text, train_labels, test_labels = train_test_split(dataSet.iloc[:,i], labelSet, test_size=0.33,random_state=23333)
        
        train_text = np.mat(train_text).T
        test_text = np.mat(test_text).T
    
        from xgboost.sklearn import XGBRegressor
        model_xgb = XGBRegressor()   
        
    
        from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
        model_rfg = RandomForestRegressor() 
        model_gb = GradientBoostingRegressor()
    
        from mlxtend.regressor import StackingRegressor    
        regressors = [model_xgb,model_rfg,model_gb]
        model = StackingRegressor(regressors=regressors, meta_regressor=model_gb)
    
        model.fit(train_text,train_labels)
        
#        preds = model.predict(train_text)
#        print('The pearsonr of training set is {}'.format(pearsonr (list(train_labels), list(preds))[0]))
#        print('The MSE of training set is {}'.format(mean_squared_error(list(train_labels), list(preds))))
          
        preds = model.predict(test_text)
#        print('The pearsonr of test set is {}'.format(pearsonr (list(test_labels), list(preds))[0]))
#        print('The MSE of test set is {}'.format(mean_squared_error(list(test_labels), list(preds))))
        V.append(pearsonr (list(test_labels), list(preds))[0])
l = ['common_words','fuzz_Qratio','fuzz_Wratio','fuzz_partial_ratio','fuzz_partial_token_set_ratio','fuzz_partial_token_sort_ratio','fuzz_token_set_ratio',	'fuzz_token_sort_ratio','tfidf_features',	'lsi_features','glove_wm','fasttex_wm'] 
plt.figure(figsize=(10,6))
plt.bar(range(1,len(V)+1),V)
plt.title('Analysis of feature weight')
plt.xticks(range(1,len(V)+1),l,size='small',rotation=30)
plt.xlabel('Features')
plt.ylabel('Weight')
plt.show()



