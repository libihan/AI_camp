import numpy as np
import pandas as pd
import nltk
from textblob import Word
from nltk import ngrams,word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.text import text_to_word_sequence

# 1）长度相关
def form_counts(essay):
    words = word_tokenize(essay) #不会自动滤除标点符号
    word_count = len(words) #单词数

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(essay.strip())
    sentence_count = len(sents) #句子数

    if sentence_count != 0:
        avg_sentence_len = word_count/sentence_count    #每个句子的平均单词数
    else:
        avg_sentence_len = 0

    character_count = len(''.join(text_to_word_sequence (essay)))   #不考虑空格和标点符号
    if word_count != 0:
        avg_word_len = character_count/word_count    #每个单词的平均字符数
    else:
        avg_word_len = 0

    exc_count = words.count('!')
    que_count = words.count('?')
    # quot_count = words.count('"')

    spelling_errors = sum([Word(word).spellcheck()[0][0] != word for word in words])   #拼写错误的单词数
    long_word = sum([len(word)>=7 for word in words])   #长单词数

    #停用词个数
    stopword = stopwords.words("english")
    stopwords_count = 0
    for word in words:
        if word in stopword:
            stopwords_count += 1

    return word_count,sentence_count,avg_sentence_len,avg_word_len,spelling_errors,long_word,exc_count,que_count,stopwords_count

# 2) language model counts
def ngrams_counts(essay):
    words = word_tokenize(essay)  #不会自动滤除标点符号
    sents = " ".join(words)

    unigrams = [grams for grams in ngrams(sents.split(), 1)]
    bigrams = [grams for grams in ngrams(sents.split(), 2)]
    trigram = [grams for grams in ngrams(sents.split(), 3)]

    unigrams_count = len(set(unigrams))
    bigrams_count = len(set(bigrams))
    trigrams_count = len(set(trigram))

    return unigrams_count,bigrams_count,trigrams_count

# 3) POS counts
def pos_counts(essay):
    noun_count,adj_count,adv_count,verb_count,fw_count = 0, 0, 0, 0, 0

    words = word_tokenize(essay)
    tags = nltk.pos_tag(words)
    for tag in tags:
        if tag[1].startswith("NN"):
            noun_count += 1
        elif tag[1].startswith("JJ"):
            adj_count += 1
        elif tag[1].startswith("RB"):
            adv_count += 1
        elif tag[1].startswith("VB"):
            verb_count += 1
        elif tag[1].startswith("FW"):
            fw_count += 1
    return noun_count,adj_count,adv_count,verb_count,fw_count

# 4) getFeature
def getFeature(essays):
    features = pd.DataFrame()

    form_features = np.array([form_counts(essay) for essay in essays])
    features['word_count'],features['sentence_count'],features['avg_sentence_len'], features['avg_word_len'],\
    features['spelling_errors'],features['long_word'],features['exc_count'],features['que_count'], features['stopwords_count']\
        = form_features[:,0],form_features[:,1],form_features[:,2],form_features[:,3],form_features[:,4],form_features[:,5],\
          form_features[:,6],form_features[:,7],form_features[:,8]

    ngrams_features = np.array([ngrams_counts(essay) for essay in essays])
    features['unigrams_count'],features['bigrams_count'],features['trigrams_count'] = ngrams_features[:,0],ngrams_features[:,1],ngrams_features[:,2]

    pos_features = np.array([pos_counts(essay) for essay in essays])
    features['noun_count'], features['adj_count'], features['adv_count'],features['verb_count'], features['fw_count'] \
        = pos_features[:,0],pos_features[:,1],pos_features[:,2],pos_features[:,3],pos_features[:,4]

    return features

if __name__ == '__main__':
    # 载入原始数据
    train_path = r'C:\Users\XPS13\Desktop\0206\AI_Camp\Week2\essay_train.txt'
    test_path = r'C:\Users\XPS13\Desktop\0206\AI_Camp\Week2\essay_test.txt'
    df_train = pd.read_table(train_path,header = None,names=["id", "text", "score1", "score2","score"])
    df_test = pd.read_table(test_path,header = None,names=["id", "text"])

    # train_features = getFeature(df_train.text)
    # train_features['score1'] = df_train.score1
    # train_features['score2'] = df_train.score2
    # train_features['score'] = df_train.score
    # train_features.to_csv('train_data.csv', index=False)

    test_features = getFeature(df_test.text)
    test_features.to_csv('test_data.csv', index=False)
