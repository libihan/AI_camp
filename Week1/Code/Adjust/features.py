# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:19:45 2018

@author: Administrator
"""
import re
import pandas as pd
import numpy as np

#==============================================================================
#  数据预处理
#==============================================================================
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def preProcess(sentence):
    # 去除标点符号
    pattern1 = '[,.:;?()!@#%$*[]=]'
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
from fuzzywuzzy import fuzz

def get_lists_intersection(s1, s2):
    s1_s2 = []
    for i in s1:
        if i in s2:
            s1_s2.append(i)
    return s1_s2

def overlap(s1_ngrams, s2_ngrams):
    s1_len = len(s1_ngrams)
    s2_len = len(s2_ngrams)
    if s1_len == 0 and s2_len == 0:
        return 0
    s1_s2_len = max(1, len(get_lists_intersection(s1_ngrams, s2_ngrams)))
    return 2 / (s1_len / s1_s2_len + s2_len / s1_s2_len)

# 此处word指单词
def get_ngram(word, n):
    ngrams = []
    word_len = len(word)
    for i in range(word_len - n + 1):
        ngrams.append(word[i: i + n])
    return ngrams

# 此处text1指句子
def get_ngram_feature(text1, text2, n):
    s1_ngrams = []
    s2_ngrams = []

    for word in text1:
        s1_ngrams.extend(get_ngram(word, n))

    for word in text2:
        s2_ngrams.extend(get_ngram(word, n))

    return overlap(s1_ngrams, s2_ngrams)


def sen2vec(model,text):
    M = []
    for s in text:
        try:
            M.append(model[s])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


def getFeature(text):
    features = pd.DataFrame()
    
    ## Basic Features
#    features['diff_len'] = text['sentence1'].apply(lambda x:len(x)) - text['sentence2'].apply(lambda x:len(x))
    
    # 共同单词个数 / 最长总单词
    s = []
    for i,j in zip(text['sentence1'],text['sentence2']):
        s.append(len(set(i).intersection(set(j))) / max(len(set(i)),len(set(j))))
    features['common_words'] = s
      
    # fuzz Features   
    features['fuzz_Qratio'] = text.apply(lambda x: fuzz.QRatio(str(x['sentence1']), str(x['sentence2'])), axis=1)
    features['fuzz_Wratio'] = text.apply(lambda x: fuzz.WRatio(str(x['sentence1']), str(x['sentence2'])), axis=1)
    features['fuzz_partial_ratio'] = text.apply(lambda x: fuzz.partial_ratio(str(x['sentence1']), str(x['sentence2'])), axis=1)
#    features['fuzz_partial_token_set_ratio'] = text.apply(lambda x: fuzz.partial_token_set_ratio(str(x['sentence1']), str(x['sentence2'])), axis=1)
    features['fuzz_partial_token_sort_ratio'] = text.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['sentence1']), str(x['sentence2'])), axis=1)
    features['fuzz_token_set_ratio'] = text.apply(lambda x: fuzz.token_set_ratio(str(x['sentence1']), str(x['sentence2'])), axis=1)
    features['fuzz_token_sort_ratio'] = text.apply(lambda x: fuzz.token_sort_ratio(str(x['sentence1']), str(x['sentence2'])), axis=1)
    
    text1 = list(text['sentence1'])
    text2 = list(text['sentence2'])
    #TFIDF
    dictionary = corpora.Dictionary(text1+text2)    #为每个出现在语料库中的单词分配了一个独一无二的整数编号
    #print(dictionary.token2id) #查看单词与编号之间的映射关系  
    corpus1 = [dictionary.doc2bow(text) for text in text1] #函数doc2bow()简单地对每个不同单词的出现次数进行了计数，并将单词转换为其编号，然后以稀疏向量的形式返回结果 
    tfidf = models.TfidfModel(corpus1)  #将bow向量中的特征进行IDF统计
    corpus1_tfidf = tfidf[corpus1]

    corpus2 = [dictionary.doc2bow(text) for text in text2]
    corpus2_tfidf = tfidf[corpus2]
    
    index = similarities.MatrixSimilarity(corpus1_tfidf) 
    sims = index[corpus2_tfidf]
    features['tfidf_features'] = np.diag(sims)
    
    # LSI模型
    lsi = models.LsiModel(corpus1_tfidf, id2word=dictionary, num_topics=len(text1))
    index = similarities.MatrixSimilarity(lsi[corpus1])
    sims = index[lsi[corpus2]]
    features['lsi_features'] = np.diag(sims)
    
    #Glove
    glove_wm = []
    gensim_file = r'C:\Users\Administrator\Desktop\AI_Camp\glove_model.txt'
    model = models.KeyedVectors.load_word2vec_format(gensim_file) 
    for sen1,sen2 in zip(text1,text2):
        glove_wm.append(model.wmdistance(sen1,sen2))
    features['glove_wm'] = glove_wm
    
    #fasttext
    fasttext = []
    gensim_file = r'C:\Users\Administrator\Desktop\AI_Camp\fasttext_model.txt'
    model = models.KeyedVectors.load_word2vec_format(gensim_file) 
    for sen1,sen2 in zip(text1,text2):
        fasttext.append(model.wmdistance(sen1,sen2))
    features['fasttex_wm'] = fasttext
    
    #ngrams
    ngram1,ngram2,ngram3 = [],[],[]
    for i,j in zip(text['sentence1'],text['sentence2']):
        ngram1.append(get_ngram_feature(i,j,1))
        ngram2.append(get_ngram_feature(i,j,2))
        ngram3.append(get_ngram_feature(i,j,3))
#    features['ngram_1'] = ngram1
    features['ngram_2'] = ngram2
    features['ngram_3'] = ngram3
         
    return features

if __name__ == '__main__':
    #==============================================================================
    # 读取数据
    #==============================================================================
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
    #==============================================================================
    # 特征提取
    #==============================================================================
    train_features = getFeature(train_text)
    train_features['score'] = train_data['score']
    train_features.to_csv('train_data.csv', index=False)
    
    test_features = getFeature(test_text)
    test_features['idx'] = train_data['score']
    test_features.to_csv('test_features.csv', index=False)