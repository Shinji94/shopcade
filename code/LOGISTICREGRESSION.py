#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 17:29:40 2018

@author: xinji
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
import os 
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
min_max_scaler = preprocessing.MinMaxScaler()



sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

os.chdir('/home/xinji/Desktop/mao')


#if you want to take a subset from whole dataset

x = pd.read_csv('full_data.csv')
list(x)
click = x[x['label'] == 1]
view = x[x['label'] == 0]
view = view.sample(frac=1).reset_index(drop=True)
temp = view[0:33724]
data = pd.concat([temp,click])
data =data.sample(frac=1).reset_index(drop=True)
data[['price']] = scaler.fit_transform(data[['price']])

del x,click,view,temp
#data.to_csv('shuffle_data.csv',index = False)
data = pd.read_csv('shuffle_data.csv')
pid = data.pid
data = data [['pid','scid','visit_hour','ts','phone_brand','brand','system','tag','ip','city','country',
         'cat1','cc','cur','merchant','price','label']]

feature = data[['label','pid','visit_hour','phone_brand','system','tag','country',
         'cat1','merchant','price','city','brand']]

one_hot = pd.read_csv('onehot_product.csv')

# =============================================================================
# Create dummy variables
# =============================================================================
shape_ = []

visit_hour = pd.get_dummies(feature['visit_hour'],prefix='visit_hour')
phone_brand = pd.get_dummies(feature['phone_brand'],prefix= 'phone_brand')
system = pd.get_dummies(feature['system'],prefix='system')
tag = pd.get_dummies(feature['tag'],prefix='tag')
city = pd.get_dummies(feature['city'],prefix='city')
country = pd.get_dummies(feature['country'],prefix='country')
features = pd.concat([feature['label'],feature['price'],pid,visit_hour,phone_brand,system,country], axis=1)      
store =features
features = pd.merge(features,one_hot,on="pid")
features = features.drop(['description','name'],axis = 1)
del visit_hour,phone_brand,system,tag,country,city
features =features.sample(frac=1).reset_index(drop=True)

X=features
X = X.drop(['label'],axis = 1)
name_space = list(features)
Y = features['label']
#import statsmodels.api as sm
#logit_model=sm.Logit(Y,X)
del data,feature
# =============================================================================
# Logistic Regression Model Fitting
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=666)
train_pid = X_train['pid']
X_train = X_train.drop(['pid'],axis = 1)
test = X_test['pid']
X_test = X_test.drop(['pid'],axis = 1)

from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import log_loss
logreg = SGDClassifier(loss="log", penalty="l1",warm_start = True)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
loss = log_loss(y_pred,y_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print('logloss of logistic regression classifier on test set: ',loss)

logreg2 = SGDClassifier(loss="log", penalty="l2",warm_start = True)
logreg2.fit(X_train, y_train)
#logreg2.partial_fit()
y_pred2 = logreg2.predict(X_test)
loss2 = log_loss(y_pred2,y_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg2.score(X_test, y_test)))
print('logloss of logistic regression classifier2 on test set: ',loss2)

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = logreg
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# =============================================================================
# roc CURVE
# =============================================================================
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

logit_roc_auc = roc_auc_score(y_test, logreg2.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg2.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression2 (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log2_ROC')
plt.show()

acc = [logreg.score(X_test, y_test),logreg2.score(X_test, y_test)]
logloss = [loss,loss2]
modelname = ['LG+L1','LG+L2']
result = pd.DataFrame({'model':modelname,'acc':acc,'logloss':logloss})
result.to_csv('model_result.csv',index = False)
# =============================================================================
# SAVE MODEL
# =============================================================================
from sklearn.externals import joblib
#joblib.dump(logit_model, 'smlogreg.pkl') 
#Later you can load back the pickled model (possibly in another Python process) with:
joblib.dump(logreg, 'logreg+L1.pkl') 
joblib.dump(logreg2, 'logreg+L2.pkl')
#clf = joblib.load('filename.pkl') 



# =============================================================================
# xgboost 
# =============================================================================



import xgboost as xgboost
from sklearn.metrics import explained_variance_score,accuracy_score,roc_auc_score
from xgboost import plot_tree,XGBClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
import math
from sklearn.model_selection import KFold
from collections import defaultdict
import operator
def ceate_feature_map(features):
    #output famp file for xgbdt for plotting and extracting feature importance
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
    
#xgboost tuning test 1 : start  from tuning the max_depth
x = [4,5,6,7,8,9,10]
aucscore = []
acc= []
rmse = []
for depth in x: 
    xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.2, gamma=0, subsample=0.75,
                               colsample_bytree=1, max_depth=depth,silent = 0,)
    traindf, testdf = train_test_split(X_train, test_size = 0.3)
    xgb.fit(X_train,y_train)
    predictions = xgb.predict(X_test)
    RMSEx = math.sqrt(np.mean(predictions - y_test)**2)
    predict = []
    for prediction in predictions:
        if prediction >0.5:
            predict.append(1)
        else:
            predict.append(0)
    accuracy = accuracy_score(y_test,predict)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('Auc Score(Train):',roc_auc_score(y_test,predictions))
    print('RMSE is : ',RMSEx)
    aucscore.append(roc_auc_score(y_test,predictions))
    acc.append(accuracy* 100.0)
    rmse.append(RMSEx)
print(aucscore)
print(acc)
print(rmse)


#xgboost tuning test 2 : start  from tuning the max_depth
x = [50,100,120,140,160,180,200,300]
aucscore = []
acc= []
rmse = []
for N in x: 
    xgb = xgboost.XGBRegressor(n_estimators=N, learning_rate=0.2, gamma=0, subsample=0.75,
                               colsample_bytree=1, max_depth=8,silent = 0,)
    traindf, testdf = train_test_split(X_train, test_size = 0.3)
    xgb.fit(X_train,y_train)
    predictions = xgb.predict(X_test)
    RMSEx = math.sqrt(np.mean(predictions - y_test)**2)
    predict = []
    for prediction in predictions:
        if prediction >0.5:
            predict.append(1)
        else:
            predict.append(0)
    accuracy = accuracy_score(y_test,predict)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('Auc Score(Train):',roc_auc_score(y_test,predictions))
    print('RMSE is : ',RMSEx)
    aucscore.append(roc_auc_score(y_test,predictions))
    acc.append(accuracy* 100.0)
    rmse.append(RMSEx)
print(aucscore)
print(acc)
print(rmse)


#xgboost tuning test 3: start  from tuning the LEARNING RATE
x = [0.01,0.05,0.1,0.15,0.2,0.2]
aucscore = []
acc= []
rmse = []
for N in x: 
    xgb = xgboost.XGBRegressor(n_estimators=N, learning_rate=0.2, gamma=0, subsample=0.75,
                               colsample_bytree=1, max_depth=8,silent = 0,)
    traindf, testdf = train_test_split(X_train, test_size = 0.3)
    xgb.fit(X_train,y_train)
    predictions = xgb.predict(X_test)
    RMSEx = math.sqrt(np.mean(predictions - y_test)**2)
    predict = []
    for prediction in predictions:
        if prediction >0.5:
            predict.append(1)
        else:
            predict.append(0)
    accuracy = accuracy_score(y_test,predict)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('Auc Score(Train):',roc_auc_score(y_test,predictions))
    print('RMSE is : ',RMSEx)
    aucscore.append(roc_auc_score(y_test,predictions))
    acc.append(accuracy* 100.0)
    rmse.append(RMSEx)
print(aucscore)
print(acc)
print(rmse)











b = xgb.get_booster()
fs = b.get_fscore()
#
#kfold = KFold(n_splits=10, random_state=7)
#results = cross_val_score(xgb, X, Y, cv=kfold)
#print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

xgboost_roc_auc = roc_auc_score(y_test, xgb.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test,  xgb.predict(X_test))
plt.figure()
plt.plot(fpr, tpr, label='xgboost (area = %0.2f)' % xgboost_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('xgb_ROC')
plt.show()

ceate_feature_map(X_train.columns)

importance = list(xgb.feature_importances_)
name_space = list(X_train)
d = dict(zip(name_space,importance))

importance = sorted(d.items(), key=operator.itemgetter(1),reverse=True)
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()
df.plot()
df[0:9].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')





# ===============
# predict mean~p
# ===============

shape_

#replace the onehot dadta in thedataset and calculate the mean ctr as product_i 's ctr
one_hot = pd.read_csv('onehot_product.csv')

for pids in one_hot.pid:
    break

temp = [pids]*len(X_test)
len(temp)


list(one_hot)[21]







