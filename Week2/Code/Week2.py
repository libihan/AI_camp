import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV,train_test_split
from scipy.stats import pearsonr

from sklearn.linear_model import LassoCV
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingRegressor



# 读取数据
train_path = r'C:\Users\XPS13\Desktop\0206\AI_Camp\Week2\essay_train.txt'
test_path = r'C:\Users\XPS13\Desktop\0206\AI_Camp\Week2\essay_test.txt'
df_train = pd.read_table(train_path, header=None, names=["id", "text", "score1", "score2", "score"])
df_test = pd.read_table(test_path, header=None, names=["id", "text"])
test_ids = df_test['id']

train_data = pd.read_csv(r'C:\Users\XPS13\Desktop\0206\AI_Camp\Feature\train_data.csv')
test_features = pd.read_csv(r'C:\Users\XPS13\Desktop\0206\AI_Camp\Feature\test_data.csv')
# train_posngs = pd.read_csv(r'C:\Users\XPS13\Desktop\0206\AI_Camp\Feature\train_posngs.csv')
# test_posngs = pd.read_csv(r'C:\Users\XPS13\Desktop\0206\AI_Camp\Feature\test_posngs.csv')
#
# train_data = pd.concat([train_posngs, train_data], axis=1)
# test_features = pd.concat([test_posngs, test_features], axis=1)


# 载入train和test数据集
dataSet = train_data.iloc[:, :-3]
labelSet = train_data.iloc[:,-3:]
train_text, test_text, train_labels, test_labels = train_test_split(dataSet, labelSet, test_size=0.2,random_state=23333)

# train_text = np.mat(train_text).T
# test_text = np.mat(test_text).T
# ==============================================================================
# 模型选择
# ==============================================================================


# ① LassoCV模型
model_lscv = LassoCV() #调参 alphas = [0.01]
# 拟合score1
model_lscv.fit(train_text, train_labels.iloc[:, -3])
lscv_pred1 = model_lscv.predict(test_text)
# 拟合score2
model_lscv.fit(train_text, train_labels.iloc[:, -2])
lscv_pred2 = model_lscv.predict(test_text)
lscv_pred = lscv_pred1 + lscv_pred2

# ② XGBRegressor模型
model_xgb1 = XGBRegressor(learning_rate =0.1, n_estimators=50, max_depth=3,min_child_weight=3, gamma=0.02,
                         colsample_bytree=0.8,subsample=0.5, seed=27)
model_xgb2 = XGBRegressor(learning_rate =0.1, n_estimators=50, max_depth=2,min_child_weight=3,gamma=0.05,
                          colsample_bytree=0.6,subsample=0.7, seed=27)
# 拟合score1
model_xgb1.fit(train_text, train_labels.iloc[:, -3])
xgb_pred1 = model_xgb1.predict(test_text)
# 拟合score2
model_xgb2.fit(train_text, train_labels.iloc[:, -2])
xgb_pred2 = model_xgb2.predict(test_text)
xgb_pred = xgb_pred1 + xgb_pred2

# ③ RandomForestRegressor模型
model_rfg1 = RandomForestRegressor(n_estimators=30, min_samples_split=90, min_samples_leaf=10,max_depth=8, random_state=10)
model_rfg2 = RandomForestRegressor(n_estimators=30, min_samples_split=90, min_samples_leaf=10,max_depth=9, random_state=10)
# 拟合score1
model_rfg1.fit(train_text, train_labels.iloc[:, -3])
rfg_pred1 = model_rfg1.predict(test_text)
# 拟合score2
model_rfg2.fit(train_text, train_labels.iloc[:, -2])
rfg_pred2 = model_rfg2.predict(test_text)
rfg_pred = rfg_pred1 + rfg_pred2

# ④ GradientBoostingRegressor模型
model_gb1 = GradientBoostingRegressor(n_estimators=40,min_samples_split=100,min_samples_leaf=20,max_depth=3, subsample=0.8,random_state=10)
model_gb2 = GradientBoostingRegressor(n_estimators=40,min_samples_split=450,min_samples_leaf=5,max_depth=4, subsample=0.8,random_state=10)
# 拟合score1
model_gb1.fit(train_text, train_labels.iloc[:, -3])
gb_pred1 = model_gb1.predict(test_text)
# 拟合score2
model_gb2.fit(train_text, train_labels.iloc[:, -2])
gb_pred2 = model_gb2.predict(test_text)
gb_pred = gb_pred1 + gb_pred2

# regressors = [model_lscv,model_xgb, model_rfg, model_gb]
# model = StackingRegressor(regressors=regressors, meta_regressor=model_xgb)

pred = (lscv_pred + xgb_pred + rfg_pred + gb_pred)/4
print(type(pred),pred.shape)

# print('Fit score1 + score2: The pearsonr of test set is {}'.format(pearsonr(list(test_labels.iloc[:, -1]), list(pred1))[0]))
print('Fit score1 + score2: the pearsonr of test set is {}'.format(pearsonr(list(test_labels.iloc[:, -1]), list(pred))[0]))

"""

model_lscv = LassoCV()
model_xgb =XGBRegressor()
model_rfg = RandomForestRegressor()
model_gb = GradientBoostingRegressor()
## 提交结果
def model_select(model):
    model.fit(train_text, train_labels.iloc[:, -3])
    preds1_test = model.predict(test_text)
    model.fit(train_text, train_labels.iloc[:, -2])
    preds2_test = model.predict(test_text)
    preds_test1 = preds1_test + preds2_test

    model.fit(train_text, train_labels.iloc[:, -1])
    preds_test2 = model.predict(test_text)
    return preds_test1,preds_test2

preds1_test1,preds1_test2 = model_select(model_lscv)
preds2_test1,preds2_test2 = model_select(model_xgb)
preds3_test1,preds3_test2 = model_select(model_rfg)
preds4_test1,preds4_test2 = model_select(model_gb)

pred1 = (preds1_test1+preds2_test1+preds3_test1+preds4_test1)/4
pred2 = (preds1_test2+preds2_test2+preds3_test2+preds4_test2)/4

print('Fit score1 + score2: The pearsonr of test set is {}'.format(pearsonr(list(test_labels.iloc[:, -1]), list(pred1))[0]))
print('Fit score: the pearsonr of test set is {}'.format(pearsonr(list(test_labels.iloc[:, -1]), list(pred2))[0]))

# with open("Week2_results", "w") as f:
#     for idx, pred in zip(test_ids, pred1):
#         f.write(str(idx) + "," + str(pred) + "\n")
# """