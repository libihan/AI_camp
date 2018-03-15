import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import CountVectorizer

## 将essay转换成稀疏矩阵

# 将词转换成tag的列表
def poslist(essay):
    tokens = word_tokenize(essay.lower().strip())
    poslist = []
    p = pos_tag(tokens)
    for (token,tag) in p:
        poslist.append(tag)

    return ' '.join(poslist)

def pos_ngs(essay):
    features = pd.DataFrame()
    # getting binarized pos-ngrams
    features['pos'] = essay.apply(poslist)  #将每句话转换成tag的列表
    cv = CountVectorizer(lowercase=False, ngram_range=(1,3), binary=True)
    ngs = cv.fit_transform(features['pos'])  #将不易操作的tags转换成词向量

    posngrams = pd.DataFrame(ngs.toarray(), index=features.index)   #Train:1200 rows x 10404 columns,共有10404个特征

    # filter ngrams that occur less than 5 times
    ngcount = posngrams.sum(axis=0).to_frame()      #[10404 rows x 1 columns]
    good_indices = ngcount[ngcount[0] >= 5].index.values
    filtered_posngs = posngrams[good_indices]       #[1200 rows x 5472 columns] 滤取出现频数大于5的特征
    # convert to string
    filtered_posngs.rename(columns = lambda x: str(x), inplace=True)    #将列名转换为str，方便索引
    # join to features and drop POS feature
    return filtered_posngs


if __name__ == '__main__':
    # 载入原始数据
    train_path = r'C:\Users\XPS13\Desktop\0206\AI_Camp\Week2\essay_train.txt'
    test_path = r'C:\Users\XPS13\Desktop\0206\AI_Camp\Week2\essay_test.txt'
    df_train = pd.read_table(train_path,header = None,names=["id", "text", "score1", "score2","score"])
    df_test = pd.read_table(test_path,header = None,names=["id", "text"])

    train_features = pd.read_csv(r'C:\Users\XPS13\Desktop\0206\AI_Camp\Week2\train_data.csv')
    test_features = pd.read_csv(r'C:\Users\XPS13\Desktop\0206\AI_Camp\Week2\test_data.csv')


    cv = CountVectorizer(lowercase=False, ngram_range=(1, 3), binary=True)
    # getting binarized pos-ngrams
    pos_train = df_train['text'].apply(poslist)  #将每句话转换成tag的列表
    ngs = cv.fit_transform(pos_train).toarray()  #将不易操作的tags转换成词向量  #Train:1200 rows x 10404 columns,共有10404个特征
    train_posngs = ngs[:, np.where(ngs.sum(axis=0) > 30)[0]]

    pos_test = df_test['text'].apply(poslist)
    ngs_test = cv.transform(pos_test).toarray()
    test_posngs = ngs_test[:, np.where(ngs.sum(axis=0) > 30)[0]]

    # feas = pd.concat([feas, filtered_posngs], axis=1)
    pd.DataFrame(train_posngs).to_csv('train_posngs.csv',index=None)
    pd.DataFrame(test_posngs).to_csv('test_posngs.csv', index=None)

