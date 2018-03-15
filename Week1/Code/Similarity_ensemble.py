# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:33:01 2018

@author: Administrator
"""
import re
import pandas as pd
import numpy as np


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
    
    #==============================================================================
    # 特征提取
    #==============================================================================
#    train_features = np.hstack(getFeature(train_text1,train_text2)) #在行上合并
#    train_text = np.mat(train_text).T
#    test_text = np.mat(test_text).T
    #==============================================================================
    # 模型训练
    #==============================================================================

#==============================================================================
#   XGBRegressor模型
#==============================================================================
    from xgboost.sklearn import XGBRegressor
#     'max_depth'--3  'min_child_weight'--1 'gamma'--0.1
#    param_test1 = {'subsample':[x/10 for x in range(1,10)],'colsample_bytree':[x/10 for x in range(1,10)]}
#    gsearch1 = GridSearchCV(estimator = XGBRegressor(learning_rate =0.1, n_estimators=120, max_depth=3,min_child_weight=1, \
#                                                     gamma=0, subsample=0.7,colsample_bytree=0.6, \
##                                                     seed=27), param_grid = param_test1,cv=5)

##    model_xgb = gsearch1
    
#    model_xgb = XGBRegressor(learning_rate =0.1, n_estimators=110, max_depth=3,min_child_weight=1, \
#                                                     gamma=0, subsample=0.7,colsample_bytree=0.6, \
#                                                     scale_pos_weight=1, seed=27)
    model_xgb = XGBRegressor()   
    
#==============================================================================
#     2) RandomForestRegressor模型
#==============================================================================
    from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
    
#    param_test1 = {'min_samples_split':range(2,10)}
#    gsearch1 = GridSearchCV(estimator = RandomForestRegressor(n_estimators= 80,max_depth=8,\
#                                                              min_samples_split=22,min_samples_leaf=2,max_features=5,\
#                                                              random_state=10), param_grid = param_test1,cv=5)
#
#    model_rfg = gsearch1    
    model_rfg = RandomForestRegressor(n_estimators= 80,max_depth=8,min_samples_split=22,min_samples_leaf=2,max_features=5,random_state=10)
#    model_rfg = RandomForestRegressor() 
#==============================================================================
#    3) GradientBoostingRegressor模型
#==============================================================================
#    param_test1 = {'n_estimators':range(10,101,10)}
#    gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators = 70,learning_rate=0.1, min_samples_split=200,max_depth=5,\
#                                                                  min_samples_leaf=20,max_features=3,subsample=0.8,random_state=10),\
#                            param_grid = param_test1, cv=5)
#    model_gb = GradientBoostingRegressor(n_estimators = 70,learning_rate=0.1, min_samples_split=200,max_depth=5,\
#                                         min_samples_leaf=20,max_features=3,subsample=0.8,random_state=10)
    model_gb = GradientBoostingRegressor()
    

#==============================================================================
#   4) LGBMRegressor模型
#==============================================================================
#    from lightgbm import LGBMRegressor
#    
#    model_lgb = LGBMRegressor()
#==============================================================================
#     5) 融合模型
#==============================================================================
#    from mlxtend.regressor import StackingRegressor    
##
#    regressors = [model_xgb,model_rfg,model_gb]
#    model = StackingRegressor(regressors=regressors, meta_regressor=model_gb)

    model = model_gb
    model.fit(train_text,train_labels)
    
#    print('The parameters of the best model are: ')
#    print(model.best_params_)
  
    preds = model.predict(train_text)
    print('The pearsonr of training set is {}'.format(pearsonr (list(train_labels), list(preds))[0]))
    print('The MSE of training set is {}'.format(mean_squared_error(list(train_labels), list(preds))))
      
    #==============================================================================
    # 预测 测试集
    #==============================================================================   
    preds = model.predict(test_text)
    
    print('The pearsonr of test set is {}'.format(pearsonr (list(test_labels), list(preds))[0]))
    print('The MSE of test set is {}'.format(mean_squared_error(list(test_labels), list(preds))))