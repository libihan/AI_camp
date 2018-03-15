# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 10:10:10 2018

@author: Administrator
"""
import pandas as pd
import matplotlib.pyplot as plt

train_path = r'C:\Users\Administrator\Desktop\AI_Camp\Tal_SenEvl_train_136KB.txt'
train_data = pd.read_table(train_path,names=["id","sent1","sent1","score"],header = None)

test_path = r'C:\Users\Administrator\Desktop\AI_Camp\Tal_SenEvl_test_62KB.txt'
test_data = pd.read_table(test_path,names=["id","sent1","sent1"],header = None)

#==============================================================================
# 载入train和test数据集
#==============================================================================
train_text = pd.DataFrame()
train_text['sentence1'] = [preProcess(x) for x in list(train_data['sent1'])]
train_text['sentence2'] = [preProcess(x) for x in list(train_data['sent1'])]

test_text = pd.DataFrame()
test_text['sentence1'] = [preProcess(x) for x in list(test_data['sent1'])]
test_text['sentence2'] = [preProcess(x) for x in list(test_data['sent1'])]

plt.figure(figsize=(20,10)) 
train_data.score.value_counts().plot(kind="bar")
plt.titel('Scores distribution')
plt.show()