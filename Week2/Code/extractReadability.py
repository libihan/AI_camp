
import pandas as pd
from textstat.textstat import textstat

if __name__ == '__main__':
    # 载入原始数据
    train_path = r'C:\Users\XPS13\Desktop\0206\AI_Camp\Week2\essay_train.txt'
    test_path = r'C:\Users\XPS13\Desktop\0206\AI_Camp\Week2\essay_test.txt'
    df_train = pd.read_table(train_path,header = None,names=["id", "text", "score1", "score2","score"])
    df_test = pd.read_table(test_path,header = None,names=["id", "text"])

    train_features = pd.read_csv(r'C:\Users\XPS13\Desktop\0206\AI_Camp\Week2\train_data.csv')
    test_features = pd.read_csv(r'C:\Users\XPS13\Desktop\0206\AI_Camp\Week2\test_data.csv')

    #可读性
    train_features['syllable_count'] = df_train.text.apply(textstat.syllable_count)
    train_features['flesch_reading_ease'] = df_train.text.apply(textstat.flesch_reading_ease)
    train_features['smog_index'] = df_train.text.apply(textstat.smog_index)
    train_features['flesch_kincaid_grade'] = df_train.text.apply(textstat.flesch_kincaid_grade)
    train_features['coleman_liau_index'] = df_train.text.apply(textstat.coleman_liau_index)
    train_features['automated_readability_index'] = df_train.text.apply(textstat.automated_readability_index)
    train_features['dale_chall_readability_score'] = df_train.text.apply(textstat.dale_chall_readability_score)
    train_features['difficult_words'] = df_train.text.apply(textstat.difficult_words)  # 困难单词数
    train_features['linsear_write_formula'] = df_train.text.apply(textstat.linsear_write_formula)
    train_features['gunning_fog'] = df_train.text.apply(textstat.gunning_fog)
    # train_features['text_standard'] = df_train.text.apply(textstat.text_standard)

    #可读性
    test_features['syllable_count'] = df_test.text.apply(textstat.syllable_count)
    test_features['flesch_reading_ease'] = df_test.text.apply(textstat.flesch_reading_ease)
    test_features['smog_index'] = df_test.text.apply(textstat.smog_index)
    test_features['flesch_kincaid_grade'] = df_test.text.apply(textstat.flesch_kincaid_grade)
    test_features['coleman_liau_index'] = df_test.text.apply(textstat.coleman_liau_index)
    test_features['automated_readability_index'] = df_test.text.apply(textstat.automated_readability_index)
    test_features['dale_chall_readability_score'] = df_test.text.apply(textstat.dale_chall_readability_score)
    test_features['difficult_words'] = df_test.text.apply(textstat.difficult_words)  # 困难单词数
    test_features['linsear_write_formula'] = df_test.text.apply(textstat.linsear_write_formula)
    test_features['gunning_fog'] = df_test.text.apply(textstat.gunning_fog)
    # test_features['text_standard'] = df_test.text.apply(textstat.text_standard)

    train_features.to_csv('train_data_R.csv', index=False)
    test_features.to_csv('test_data_R.csv', index=False)
