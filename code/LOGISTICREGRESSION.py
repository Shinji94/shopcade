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
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

os.chdir('/home/xinji/Desktop/mao')
data = pd.read_csv('data.csv')
pid = data.pid
data = data [['scid','visit_hour','ts','phone_brand','system','tag','ip','city','country',
         'cat1','cc','cur','merchant','price','label']]
y = data['label']

feature = data[['visit_hour','phone_brand','system','tag','city','country',
         'cat1','cc','cur','merchant','price']]

# =============================================================================
# Create dummy variables
# =============================================================================
shape_ = []

visit_hour = pd.get_dummies(feature['visit_hour'])
phone_brand = pd.get_dummies(feature['phone_brand'])
system = pd.get_dummies(feature['system'])
tag = pd.get_dummies(feature['tag'])
city = pd.get_dummies(feature['city'])
country = pd.get_dummies(feature['country'])
cat1 = pd.get_dummies(feature['cat1'])
cc = pd.get_dummies(feature['cc'])
cur = pd.get_dummies(feature['cur'])
merchant = pd.get_dummies(feature['merchant'])
features = pd.concat([feature['price'], visit_hour,phone_brand,system,tag,cat1,cc,cur,merchant,country,city], axis=1)      

shape_=[1,visit_hour.shape[1],phone_brand.shape[1],system.shape[1],tag.shape[1],
        cat1.shape[1],cc.shape[1],cur.shape[1],merchant.shape[1],country.shape[1],country.shape[1],city.shape[1]]

del visit_hour,phone_brand,system,tag,city,country,cat1,cc,cur,merchant

X=features

import statsmodels.api as sm

Y = y.astype('bool')
logit_model=sm.Logit(Y,X)

# =============================================================================
# Logistic Regression Model Fitting
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=666)
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import log_loss
logreg = SGDClassifier(loss="log", penalty="l1")
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
loss = log_loss(y_pred,y_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print('Accuracy of logistic regression classifier on test set: ',loss)


logreg2 = SGDClassifier(loss="log", penalty="l2")
logreg2.fit(X_train, y_train)

y_pred2 = logreg2.predict(X_test)
loss2 = log_loss(y_pred2,y_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg2.score(X_test, y_test)))
print('Accuracy of logistic regression classifier on test set: ',loss2)

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
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
# predicting 
# =============================================================================


