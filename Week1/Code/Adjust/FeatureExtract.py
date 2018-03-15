from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus.reader import WordNetError
from sklearn.metrics import r2_score, make_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from nltk.corpus import brown, wordnet
from scipy.stats import pearsonr, spearmanr
from matplotlib import pyplot as plt
from Preprocessing import loadfile
import nltk
import numpy as np
import pickle
import nltk
import numpy as np
import re
import os
import seaborn as sns
import pandas as pd
# nert_tagger = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
# pos_tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')
# parser = StanfordParser('edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')

def vertorlize(content):
    vectorizer = CountVectorizer()
    X = vectorizer.fit(content)
    return X.toarray()


def bag_of_words(sen1,sen2):
    return  cosine_similarity([sen1],[sen2])[0][0]

def topic_id(all_sens):
    lda = LatentDirichletAllocation(n_topics=6,
                                    learning_offset=50.,
                                    random_state=0)
    docres = lda.fit_transform(all_sens)
    return docres

def tf_idf(sen1,sen2):
#     print("s1:",sen1,"s2:",sen2)
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform([sen1,sen2]).toarray()
    return tf_idf[0], tf_idf[1]


def lcs_dp(input_x, input_y):
    # input_y as column, input_x as row
    dp = [([0] * len(input_y)) for i in range(len(input_x))]
    maxlen = maxindex = 0
    for i in range(0, len(input_x)):
        for j in range(0, len(input_y)):
            if input_x[i] == input_y[j]:
                if i != 0 and j != 0:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                if i == 0 or j == 0:
                    dp[i][j] = 1
                if dp[i][j] > maxlen:
                    maxlen = dp[i][j]
                    maxindex = i + 1 - maxlen
                    # print('最长公共子串的长度是:%s' % maxlen)
                    # print('最长公共子串是:%s' % input_x[maxindex:maxindex + maxlen])
    return maxlen,input_x[maxindex:maxindex + maxlen]

def not_empty(s):
    return s and s.strip()


def tag_and_parser(str_sen1,str_sen2):
#     print(str_sen1.split(" "),str_sen2)
    sen1 = list(filter(not_empty, str_sen1.split(" ")))
    sen2 = list(filter(not_empty, str_sen2.split(" ")))
    # print("sen1:",sen1)
    post_sen1 = nltk.pos_tag(sen1)
    post_sen2 = nltk.pos_tag(sen2)
    pos1 ,pos2 = "", ""
    # print(post_sen1,post_sen2)
    for word,pos in post_sen1:
        pos1 += pos+" "
    for word,pos in post_sen2:
        pos2 += pos+" "
    # print(pos1,pos2)
    maxlen, subseq = lcs_dp(pos1,pos2)
    return  len(subseq.split(" "))/len(str_sen1.split(' ')), len(subseq.split(" "))/len(str_sen2.split(' '))


def all_features(content):
    vectorlize = CountVectorizer()
    sents = []
    sents1 = []
    sents2 = []
    scores = []
    for line in content[:-1]:
    #         print(line[1])
        sents1.append(line[1])
        sents2.append(line[2])
        sents.append(line[1])
        sents.append(line[2])
        if len(line) > 3:
            scores.append(float(line[3]))

    Sents = vectorlize.fit_transform(sents).toarray()
    Sents1 = vectorlize.transform(sents1).toarray()
    Sents2 = vectorlize.transform(sents2).toarray()
    with open("model.pickle","wb") as f:
        pickle.dump(vectorlize, f)
    tfidf_Sents1 = []
    tfidf_Sents2 = []
    tfidf_Sents = []
    cosine = []
    pos_lcs = []
    for i in range(len(sents1)):
        tfidf_Sent1, tfidf_Sent2 = tf_idf(Sents1[i],Sents2[i])
        tfidf_Sents1.append(tfidf_Sent1)
        tfidf_Sents2.append(tfidf_Sent2)
        tfidf_Sents.append(tfidf_Sent1)
        tfidf_Sents.append(tfidf_Sent2)
        cosine.append(cosine_similarity([tfidf_Sent1],[tfidf_Sent2])[0][0])
        lcs1, lcs2 = tag_and_parser(sents1[i],sents2[i])
        pos_lcs.append([lcs1, lcs2])
    tp_Sents = topic_id(tfidf_Sents)
    return cosine, pos_lcs,tfidf_Sents1,tfidf_Sents2,tp_Sents,scores



wordnet_lemmatizer = WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words("english"))

tagger = nltk.tag.pos_tag
frequency_list = FreqDist(i.lower() for i in brown.words())
all_words_count = 0
for i in frequency_list:
    all_words_count += frequency_list[i]


def get_words(sentence):
    return [i.strip('., ') for i in sentence.split(' ')]


with open('word_to_vec.txt', 'r') as f:
    embeddings = {}
    for line in f.readlines():
        args = get_words(line.strip("\n\t "))
        embeddings[args[0]] = [float(i) for i in args[1:]]

def get_ngram(word, n):
    ngrams = []
    word_len = len(word)
    for i in range(word_len - n + 1):
        ngrams.append(word[i: i + n])
    return ngrams


def get_lists_intersection(s1, s2):
    s1_s2 = []
    for i in s1:
        if i in s2:
            s1_s2.append(i)
    return s1_s2


def overlap(sentence1_ngrams, sentence2_ngrams):
    s1_len = len(sentence1_ngrams)
    s2_len = len(sentence2_ngrams)
    if s1_len == 0 and s2_len == 0:
        return 0
    s1_s2_len = max(1, len(get_lists_intersection(sentence2_ngrams, sentence1_ngrams)))
    return 2 / (s1_len / s1_s2_len + s2_len / s1_s2_len)


def get_ngram_feature(sentence1, sentence2, n):
    sentence1_ngrams = []
    sentence2_ngrams = []

    for word in sentence1:
        sentence1_ngrams.extend(get_ngram(word, n))

    for word in sentence2:
        sentence2_ngrams.extend(get_ngram(word, n))

    return overlap(sentence1_ngrams, sentence2_ngrams)
def is_subset(s1, s2):
    for i in s1:
        if i not in s2:
            return False
    return True


def get_numbers_feature(sentence1, sentence2):
    s1_numbers = [float(i) for i in re.findall(r"[-+]?\d+\.?\d*", " ".join(sentence1))]
    s2_numbers = [float(i) for i in re.findall(r"[-+]?\d+\.?\d*", " ".join(sentence2))]
    s1_s2_numbers = []
    for i in s1_numbers:
        if i in s2_numbers:
            s1_s2_numbers.append(i)

    s1ands2 = max(len(s1_numbers) + len(s2_numbers), 1)
    return [np.log(1 + s1ands2), 2 * len(s1_s2_numbers) / s1ands2,
            is_subset(s1_numbers, s2_numbers) or is_subset(s2_numbers, s1_numbers)]


def get_shallow_features(sentence):
    counter = 0
    for word in sentence:
        if len(word) > 1 and (re.match("[A-Z].*]", word) or re.match("\.[A-Z]+]", word)):
            counter += 1
    return counter


def get_word_embedding(inf_content, word):
    if inf_content:
        return np.multiply(information_content(word), embeddings.get(word, np.zeros(300)))
    else:
        return embeddings.get(word, np.zeros(300))


def sum_embeddings(words, inf_content):
    vec = get_word_embedding(inf_content, words[0])
    for word in words[1:]:
        vec = np.add(vec, get_word_embedding(inf_content, word))
    return vec


def word_embeddings_feature(sentence1, sentence2):
    return cosine_similarity(unpack(sum_embeddings(sentence1, False)),
                             unpack(sum_embeddings(sentence2, False)))[0][0]


def information_content(word):
    return np.log(all_words_count / max(1, frequency_list[word]))


def unpack(param):
    return param.reshape(1, -1)


def weighted_word_embeddings_feature(sentence1, sentence2):
    return cosine_similarity(unpack(sum_embeddings(sentence1, True)),
                             unpack(sum_embeddings(sentence2, True)))[0][0]


def weighted_word_coverage(s1, s2):
    s1_s2 = get_lists_intersection(s1, s2)
    return np.sum([information_content(i) for i in s1_s2]) / np.sum([information_content(i) for i in s2])


def harmonic_mean(s1, s2):
    if s1 == 0 or s2 == 0:
        return 0
    return s1 * s2 / (s1 + s2)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('A') or treebank_tag.startswith('JJ'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def get_synset(word):
    try:
        return wordnet.synset(word + "." + get_wordnet_pos(tagger([word])[0][1]) + ".01")
    except :
        return 0


def wordnet_score(word, s2):
    if word in s2:
        return 1
    else:
        similarities = []
        for w in s2:
            try:
                value = get_synset(word).path_similarity(get_synset(w))
                if value is None:
                    value = 0
                similarities.append(value)
            except AttributeError:
                similarities.append(0)
        return np.max(similarities)


def wordnet_overlap(s1, s2):
    suma = 0
    for w in s1:
        suma += wordnet_score(w, s2)
    return suma / len(s2)


def feature_vector(a, b):
    fvec = []
    # Ngram overlap

    fvec.append(get_ngram_feature(a, b, 1))
    fvec.append(get_ngram_feature(a, b, 2))
    fvec.append(get_ngram_feature(a, b, 3))

    # WordNet-aug. overlap -
    fvec.append(harmonic_mean(wordnet_overlap(a, b), wordnet_overlap(b, a)))

    # Weighted word overlap -
    fvec.append(harmonic_mean(weighted_word_coverage(a, b),
                              weighted_word_coverage(b, a)))
    # sentence num_of_words differences -
    fvec.append(abs(len(a) - len(b)))

    # summed word embeddings - lagano
    fvec.append(word_embeddings_feature(a, b))
    fvec.append(weighted_word_embeddings_feature(a, b))

    # Shallow NERC - lagano
    fvec.append(get_shallow_features(a))
    fvec.append(get_shallow_features(b))

    # Numbers overlap - returns list of 3 features
    fvec.extend(get_numbers_feature(a, b))
    return fvec



def F_vec(contents):
    train_vec = []
    for line in contents:
        if len(line) <2:
            break
        train_vec.append(np.array(feature_vector(line[1], line[2]), dtype=np.float64))
    return train_vec


def extract():
    scaler = StandardScaler()
    content1 = loadfile('train.txt')
    train_vec = F_vec(content1[:-1])
    cosine, pos_lcs, tfidf_Sents1, tfidf_Sents2, tp_Sents, scores = all_features(content1)
    tp_Sents = np.array(tp_Sents)
    cosine = np.array(cosine)
    pos_lcs = np.array(pos_lcs)
    tp_Sents = np.array(tp_Sents)
    X_train = np.c_[cosine,pos_lcs,tp_Sents[::2],tp_Sents[1::2],train_vec]
    X_train = scaler.fit_transform(X_train)
    Y_train = np.array(scores)

    content2 = loadfile("test.txt")
    test_vec = F_vec(content2)
    Y_ids = []
    for line in content2:
        if len(line) <2:
            break
        Y_ids.append(line[0])
    cosine, pos_lcs, tfidf_Sents1, tfidf_Sents2, tp_Sents, scores = all_features(content2)
    tp_Sents = np.array(tp_Sents)
    cosine = np.array(cosine)
    pos_lcs = np.array(pos_lcs)
    tp_Sents = np.array(tp_Sents)
    X_test = np.c_[cosine, pos_lcs, tp_Sents[::2], tp_Sents[1::2],test_vec]
    X_test = scaler.transform(X_test)
    with open("data.pickle","wb") as f:
        pickle.dump([X_train,Y_train, X_test, Y_ids], f)
    return X_train, Y_train, X_test, Y_ids
extract()
