# 单个特征作用于模型后,pearsonr得分
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,train_test_split
from scipy.stats import pearsonr

from sklearn.linear_model import LassoCV
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

def model_select(model):
    model.fit(train_text, train_labels.iloc[:, -3])
    preds1_test = model.predict(test_text)
    model.fit(train_text, train_labels.iloc[:, -2])
    preds2_test = model.predict(test_text)
    preds_test1 = preds1_test + preds2_test

    model.fit(train_text, train_labels.iloc[:, -1])
    preds_test2 = model.predict(test_text)
    return preds_test1,preds_test2

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()-rect.get_width(), 1.03*height, '%s' % round(height,2))

if __name__ == '__main__':
    # 读取数据
    train_path = r'C:\Users\XPS13\Desktop\0206\AI_Camp\Week2\essay_train.txt'
    test_path = r'C:\Users\XPS13\Desktop\0206\AI_Camp\Week2\essay_test.txt'
    df_train = pd.read_table(train_path, header=None, names=["id", "text", "score1", "score2", "score"])
    df_test = pd.read_table(test_path, header=None, names=["id", "text"])
    test_ids = df_test['id']

    train_data = pd.read_csv(r'C:\Users\XPS13\Desktop\0206\AI_Camp\Feature\train_posngs.csv')
    test_features = pd.read_csv(r'C:\Users\XPS13\Desktop\0206\AI_Camp\Feature\test_posngs.csv')

    # 载入train和test数据集
    dataSet = train_data.iloc[:, :-3]
    labelSet = train_data.iloc[:,-3:]

    y_label = []
    for i in range(test_features.shape[1]):
        train_text, test_text, train_labels, test_labels = train_test_split(dataSet.iloc[:,i], labelSet, test_size=0.33,random_state=23333)

        train_text = np.mat(train_text).T
        test_text = np.mat(test_text).T
        # ==============================================================================
        # 模型选择
        # ==============================================================================
        model_lscv = LassoCV()
        model_xgb = XGBRegressor()
        model_rfg = RandomForestRegressor()
        model_gb = GradientBoostingRegressor()

        preds1_test1,preds1_test2 = model_select(model_lscv)
        preds2_test1,preds2_test2 = model_select(model_xgb)
        preds3_test1,preds3_test2 = model_select(model_rfg)
        preds4_test1,preds4_test2 = model_select(model_gb)

        pred1 = (preds1_test1+preds2_test1+preds3_test1+preds4_test1)/4
        pred2 = (preds1_test2+preds2_test2+preds3_test2+preds4_test2)/4
        print('-----第%s个特征-------'%i)
        print('Fit score1 + score2: The pearsonr of test set is {}'.format(pearsonr(list(test_labels.iloc[:, -1]), list(pred1))[0]))
        print('Only fit score: the pearsonr of test set is {}'.format(pearsonr(list(test_labels.iloc[:, -1]), list(pred2))[0]))

        y_label.append(pearsonr(list(test_labels.iloc[:, -1]), list(pred1))[0])

    plt.figure(figsize=(70, 20))
    # plt.xticks(range(len(y_label)),dataSet.columns,size='small',rotation=30)
    rect = plt.bar(range(len(y_label)),y_label,width = 0.35)
    plt.ylim(0, 0.9)

    autolabel(rect)
    plt.show()