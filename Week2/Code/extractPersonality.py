import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def sentiment_tagger(sid,sent_detector,essay):
    neg_sentiment,neu_sentiment,pos_sentiment = 0, 0 ,0 # 初始化neg,neu和pos的语气的句子数为0
    sents = sent_detector.tokenize(essay.strip())   # 分句

    for sent in sents:
        ss = sid.polarity_scores(sent)  # 判断分句的情感语气
        for k in sorted(ss):    # 此处排序没用，只是对键排序？
            if k == 'neg':
                neg_sentiment += ss[k]	#全句的情感语气 = 分句的和
            elif k == 'neu':
                neu_sentiment += ss[k]
            elif k == 'pos':
                pos_sentiment += ss[k]
    return neg_sentiment,neu_sentiment,pos_sentiment

if __name__ == '__main__':
    # 载入原始数据
    train_path = r'C:\Users\XPS13\Desktop\0206\AI_Camp\Week2\essay_train.txt'
    test_path = r'C:\Users\XPS13\Desktop\0206\AI_Camp\Week2\essay_test.txt'
    df_train = pd.read_table(train_path,header = None,names=["id", "text", "score1", "score2","score"])
    df_test = pd.read_table(test_path,header = None,names=["id", "text"])

    sid = SentimentIntensityAnalyzer()
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    train_features = pd.DataFrame()
    sentiment_features = np.array([sentiment_tagger(sid,sent_detector,essay) for essay in df_train.text])
    train_features['neg_sentiment'], train_features['neu_sentiment'], train_features['pos_sentiment'] = \
        sentiment_features[:,0], sentiment_features[:,1], sentiment_features[:,2]
    train_features.to_csv('train_data_P.csv', index=False)

    test_features = pd.DataFrame()
    sentiment_features = np.array([sentiment_tagger(sid,sent_detector,essay) for essay in df_test.text])
    test_features['neg_sentiment'], test_features['neu_sentiment'], test_features['pos_sentiment'] = \
        sentiment_features[:,0], sentiment_features[:,1], sentiment_features[:,2]
    test_features.to_csv('test_data_P.csv', index=False)
