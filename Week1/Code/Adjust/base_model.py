# 传统机器学习所使用的模型
class model:
    from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    def svr(self, c=5, g = 3):
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler
        import scipy as sc
        #  索引最好的c = 5 gamma =1
        # bag of words c = le3 gamma = 0.1
        # tfidf c=2.8, g=2
        lasso = make_pipeline(SVR(kernel='rbf', C=c, gamma=g))
        lasso.fit(self.X_train,self.y_train)
        print("the model is SVR and the test's pearsonr is: ", sc.stats.pearsonr(self.y_test, lasso.predict(self.X_test))[0])
        return lasso
    def rf(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.ensemble import RandomForestRegressor
        import scipy as sc
        rf = make_pipeline(RandomForestRegressor(random_state=590,n_estimators =6))
        rf.fit(self.X_train,self.y_train)
        print("the model is rf and the test's pearsonr is: ", sc.stats.pearsonr(self.y_test, rf.predict(self.X_test))[0])
        return rf
    def gboost(self):
        from sklearn.ensemble import GradientBoostingRegressor
        import scipy as sc
        GBoost = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01,
                                           max_depth=12, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=97,
                                           loss='ls', random_state =200)
        GBoost.fit(self.X_train,self.y_train)
        print("the model is gboost and the test's pearsonr is: ", sc.stats.pearsonr(self.y_test, GBoost.predict(self.X_test))[0])
        return GBoost
    def xgboost(self):
        from xgboost import XGBRegressor
        import xgboost as xgb
        import scipy as sc
        #     for i in range(0,1000,10):
        model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=10,
                                     learning_rate=0.01, max_depth=11,
                                     min_child_weight=1.7817, n_estimators=500,
                                     reg_alpha=0.01, reg_lambda=5,
                                     subsample=0.5213, silent=1,
                                     seed =1024, nthread = -1)
        model_xgb.fit(self.X_train,self.y_train)
        print("the model is xgboost and the test's pearsonr is: ", sc.stats.pearsonr(self.y_test, model_xgb.predict(self.X_test))[0])
        return model_xgb
    def lgb(self):
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
        import scipy as sc
        model_lgb = LGBMRegressor(objective='regression',num_leaves=5,
                                  learning_rate=0.05, n_estimators=550,
                                  max_bin = 25, bagging_fraction = 1,
                                  bagging_freq = 5, feature_fraction = 0.7,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf =42, min_sum_hessian_in_leaf = 40)
        model_lgb.fit(self.X_train,self.y_train)
        print("the model is lgb and the test's pearsonr is: ", sc.stats.pearsonr(self.y_test, model_lgb.predict(self.X_test))[0])
        return model_lgb
    def stacking(self):
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler,MinMaxScaler
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
        from xgboost import XGBRegressor
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
        import xgboost as xgb
        from mlxtend.regressor import StackingRegressor
        import scipy as sc
        lasso = make_pipeline(SVR(kernel='rbf', C=2.8, gamma=2))
        rf = make_pipeline(RandomForestRegressor(random_state=590,n_estimators =6))
        GBoost = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01,
                                           max_depth=12, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=97,
                                           loss='ls', random_state =200)
        model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=10,
                                     learning_rate=0.01, max_depth=11,
                                     min_child_weight=1.7817, n_estimators=500,
                                     reg_alpha=0.01, reg_lambda=5,
                                     subsample=0.5213, silent=1,
                                     seed =1024, nthread = -1)
        model_lgb = LGBMRegressor(objective='regression',num_leaves=5,
                                  learning_rate=0.05, n_estimators=550,
                                  max_bin = 25, bagging_fraction = 1,
                                  bagging_freq = 5, feature_fraction = 0.7,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf =42, min_sum_hessian_in_leaf = 40)
        regressors = [rf,lasso,GBoost, model_lgb,model_xgb]
        stregr = StackingRegressor(regressors=regressors, meta_regressor=model_xgb)
        stregr.fit(self.X_train,self.y_train)
        print("the model is staking and the test's pearsonr is: ", sc.stats.pearsonr(self.y_test, stregr.predict(self.X_test))[0])
        return stregr
    class _AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

        def __init__(self, models):
            self.models = models
        # we define clones of the original models to fit the data in
        def fit(self, X, y):
            from sklearn.base import clone
            self.models_ = [clone(x) for x in self.models]
            # Train cloned base models
            for model in self.models_:
                model.fit(X, y)
            return self
        #Now we do the predictions for cloned models and average them
        def predict(self, X):
            predictions = np.column_stack([
                model.predict(X) for model in self.models_
            ])
            return np.mean(predictions, axis=1)
    def average(self):
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler,MinMaxScaler
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
        from xgboost import XGBRegressor
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
        import xgboost as xgb
        from mlxtend.regressor import StackingRegressor
        from sklearn.kernel_ridge import KernelRidge
        import scipy as sc
        #         self._load_package()
        # c=7,g=0.075
        lasso = make_pipeline(SVR(kernel='rbf', C=2.8, gamma=2))
        rf = make_pipeline(RandomForestRegressor(random_state=590,n_estimators =6))
        GBoost = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01,
                                           max_depth=12, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=97,
                                           loss='ls', random_state =200)
        model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=10,
                                     learning_rate=0.01, max_depth=11,
                                     min_child_weight=1.7817, n_estimators=500,
                                     reg_alpha=0.01, reg_lambda=5,
                                     subsample=0.5213, silent=1,
                                     seed =1024, nthread = -1)
        model_lgb = LGBMRegressor(objective='regression',num_leaves=5,
                                  learning_rate=0.05, n_estimators=550,
                                  max_bin = 25, bagging_fraction = 1,
                                  bagging_freq = 5, feature_fraction = 0.7,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf =42, min_sum_hessian_in_leaf = 40)
        regressors = [rf, lasso,GBoost, model_lgb,model_xgb]
        stregr = StackingRegressor(regressors=regressors, meta_regressor=model_xgb)
        averaged_models =self._AveragingModels(models = (rf,stregr,model_xgb,lasso))
        averaged_models.fit(self.X_train,self.y_train)
        print("the model is average and the test's pearsonr is: ", sc.stats.pearsonr(self.y_test, averaged_models.predict(self.X_test))[0])
        return averaged_models

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def cv(data):
    count_vectorizer = CountVectorizer()

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer

trains = pd.read_table("/home/kesci/input/Education_NLP/Tal_SenEvl_train_136KB.txt",names=["id","s1","s2","score"],sep="\t")
tests = pd.read_table("/home/kesci/input/Education_NLP/Tal_SenEvl_test_62KB.txt",names=["id",'s1','s2'],sep="\t")
data = pd.concat([trains, tests],axis=0)
bow_data = data[["id","s1","s2","score"]]
bow_data["words"] = bow_data["s1"] + bow_data["s2"]
bow_test = bow_data[bow_data.score.isnull()]
bow_train = bow_data[~bow_data.score.isnull()]
list_test = bow_test["words"].tolist()
list_corpus = bow_train["words"].tolist()
list_labels = bow_train["score"].tolist()
# list_corpus
X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, random_state=42)
X_train_counts, count_vectorizer = cv(X_train)
X_test_counts = count_vectorizer.transform(X_test)
test_counts = count_vectorizer.transform(list_test)
print(X_train_counts.shape, X_test_counts.shape, test_counts.shape)
# print(X_train_counts)
# m = model(X_train_counts, X_test_counts, y_train, y_test)
# m.svr(c=7,g=0.075)
# m.rf()
# m1 = model(X_train_counts.toarray(), X_test_counts.toarray(), y_train, y_test)
# m1.xgboost()
# m1.lgb()
# m1.gboost()
# average = m1.average()
# stacking = m1.stacking()
# test_pd["score"] = average.predict(test_counts.toarray())
# test_pd["score"] = test_pd["score"].apply(lambda x: 0 if x < 0 else (5 if x >5 else x))
# test_pd["score"].describe()
# test_pd[["id","score"]].to_csv("submission_sample",index=False, header=False)
# !pwd && ls && head -n 5 submission_sample
# !wget -nv -O kesci_submit https://cdn.kesci.com/submit_tool/v1/kesci_submit&&chmod +x kesci_submit
# !./kesci_submit -token 你的token号 -file submission_sample