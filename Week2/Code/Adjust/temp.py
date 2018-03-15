from __future__ import print_function

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

from keras.preprocessing.text import text_to_word_sequence

BASE_DIR = ''
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# 1) 转换成词向量
def create_lookup_table(filename):
    """
    :param filename:word2vec文件
    :return:字典{word,向量}
    """
    embeddings_index = {}
    with open(filename) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

lookup_table = create_lookup_table(r'C:\Users\Administrator\Desktop\AI_Camp\Week1\simple.glove.840B.300d.txt')

def sent2vec(text):
    """
    :param text: 句子
    :return: 词向量
    """
    words = [w for w in text_to_word_sequence(text)]

    M = []
    for w in words:
        try:
            M.append(lookup_table[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)

    return v / np.sqrt((v ** 2).sum())

# 2)
train_path = r'C:\Users\Administrator\Desktop\AI_Camp\Week2\essay_train.txt'
test_path = r'C:\Users\Administrator\Desktop\AI_Camp\Week2\essay_test.txt'
df_train = pd.read_table(train_path,header = None,names=["id", "text", "score1", "score2","score"])
df_test = pd.read_table(test_path,header = None,names=["id", "text"])
# 转换成词向量
X_train = df_train.text.apply(sent2vec)
y_train = df_train.text


# 1) 转换成词向量
def create_lookup_table(filename):
    """
    :param filename:word2vec文件
    :return:字典{word,向量}
    """
    embeddings_index = {}
    with open(filename) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

lookup_table = create_lookup_table(r'C:\Users\Administrator\Desktop\AI_Camp\Week1\simple.glove.840B.300d.txt')

def sent2vec(text):
    """
    :param text: 句子
    :return: 词向量
    """
    words = [w for w in text_to_word_sequence(text)]

    M = []
    for w in words:
        try:
            M.append(lookup_table[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)

    return v / np.sqrt((v ** 2).sum())

# 2)
train_path = r'C:\Users\Administrator\Desktop\AI_Camp\Week2\essay_train.txt'
test_path = r'C:\Users\Administrator\Desktop\AI_Camp\Week2\essay_test.txt'
df_train = pd.read_table(train_path,header = None,names=["id", "text", "score1", "score2","score"])
df_test = pd.read_table(test_path,header = None,names=["id", "text"])
# 转换成词向量
X_train = df_train.text.apply(sent2vec)
y_train = df_train.text

text = pd.concat([df_train.text,df_test.text]).tolist()
word_sequences = [text_to_word_sequence(s) for s in text]

# 3) vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(df_train.text)
sequences = tokenizer.texts_to_sequences(df_train.text)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# 4) split the data into a training set and a validation set
indices = np.arange(df_train.shape[0])
np.random.shuffle(indices)
df_train = df_train[indices]
df_score1 = df_train.score1[indices]
df_score2 = df_train.score2[indices]

num_validation_samples = int(VALIDATION_SPLIT * df_train.shape[0])

x_train = df_train[:-num_validation_samples]
y_train = df_score1[:-num_validation_samples]
x_val = df_train[-num_validation_samples:]
y_val = df_score1[-num_validation_samples:]

print('Preparing embedding matrix.')

# 5) prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))


def get_sat_word(path):
    with open(path, 'r') as f:
        sat = f.read()
    sat_words = []
    for sentence in sat.split('\n'):
        if len(sentence) < 1:
            continue
        sat_words.append(sentence.split()[0])
    return sat_words


# 1）长度相关
def form_counts(content):
    """
    :param sentence:
    :return: 单词数word_count,句子数sentence_count,每个句子的平均单词数avg_sentence_len
              拼写错误单词数spelling_errors,长单词数long_word
    """
    char_count = len(content)
    words = nltk.word_tokenize(content.strip())  # 会自动滤除标点符号/lower
    word_count = len(words)  # 单词数

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(content.strip())
    sentence_count = len(sents)  # 句子数

    if sentence_count != 0:
        avg_sentence_len = word_count / sentence_count  # 每个句子的平均单词数
    else:
        avg_sentence_len = 0

    spelling_errors = sum([Word(word).spellcheck()[0][0] != word for word in words])  # 拼写错误的单词数
    long_word = sum([len(word) >= 7 for word in words])  # 长单词数

    exc_count = words.count('!')
    que_count = words.count('?')

    stopwords = set(nltk.corpus.stopwords.words("english"))
    sat_words = get_sat_word('SAT_words')
    stopwords_num = 0
    sat_num = 0
    for word in words:
        if word in stopwords:
            stopwords_num += 1
        if word in sat_words:
            sat_num += 1

    return char_count, word_count, sentence_count, avg_sentence_len, spelling_errors, \
           long_word, exc_count, que_count, stopwords_num, sat_num


# 2) language model counts
def ngrams_counts(content):
    words = nltk.word_tokenize(content.strip())  # 会自动滤除标点符号/lower
    sents = " ".join(words)

    unigrams = [grams for grams in ngrams(sents.split(), 1)]
    bigrams = [grams for grams in ngrams(sents.split(), 2)]
    trigram = [grams for grams in ngrams(sents.split(), 3)]

    unigrams_count = len([(item[0], unigrams.count(item)) for item in sorted(set(unigrams))])
    bigrams_count = len([(item, bigrams.count(item)) for item in sorted(set(bigrams))])
    trigrams_count = len([(item, trigram.count(item)) for item in sorted(set(trigram))])

    return unigrams_count, bigrams_count, trigrams_count


# 3) POS counts
def pos_counts(content):
    noun_count, adj_count, adv_count, verb_count, fw_count = 0, 0, 0, 0, 0

    words = nltk.word_tokenize(content)
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
    return noun_count, adj_count, adv_count, verb_count, fw_count


# 4) getFeature
def get_dataframe(content):
    features = pd.DataFrame()

    features['word_count'], features['sentence_count'], features['avg_sentence_len'], \
    features['spelling_errors'], features['long_word'] = form_counts(content)

    features['unigrams_count'], features['bigrams_count'], features['trigrams_count'] = ngrams_counts(content)
    features['noun_count'], features['adj_count'], features['adv_count'], features['verb_count'], features[
        'fw_count'] = pos_counts(content)

    return features


def get_features(contents):
    features = []
    for content in contents:
        char_count, word_count, sentence_count, avg_sentence_len, spelling_errors, \
        long_word, exc_count, que_count, stopwords_num, sat_num = form_counts(content)
        unigrams_count, bigrams_count, trigrams_count = ngrams_counts(content)
        noun_count, adj_count, adv_count, verb_count, fw_count = pos_counts(content)
        feature = np.c_[char_count, word_count, sentence_count, avg_sentence_len, spelling_errors,
                        long_word, exc_count, que_count, stopwords_num, sat_num, unigrams_count,
                        bigrams_count, trigrams_count, noun_count, adj_count, adv_count, verb_count, fw_count]
        features.append(feature)
    return np.array(features)



from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import CountVectorizer

def poslist(essay):
    tokens = word_tokenize(essay.lower().strip())
    poslist = []
    p = pos_tag(tokens)
    for (token,tag) in p:
        poslist.append(tag)

    return ' '.join(poslist)

# getting binarized pos-ngrams
feas['pos'] = feas['essay'].apply(poslist)
cv = CountVectorizer(lowercase=False, ngram_range=(1,3), binary=True)
ngs = cv.fit_transform(feas['pos'])

posngrams = pd.DataFrame(ngs.toarray(), index=feas.index)

# filter ngrams that occur less than 5 times
ngcount = posngrams.sum(axis=0).to_frame()
good_indices = ngcount[ngcount[0]>=5].index.values
filtered_posngs = posngrams[good_indices]
# convert to string
filtered_posngs.rename(columns = lambda x: str(x), inplace=True)
# join to features and drop POS feature


feas = pd.concat([feas, filtered_posngs], axis=1)
feas.drop('pos', axis=1, inplace=True)




