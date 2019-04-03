import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
import  matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
import jieba
import jieba.analyse
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import datetime
import gensim

def c(x):
    if x==0:
        return 1
    else:
        return 0


starttime = datetime.datetime.now()
train = pd.read_csv('F:\\pycharm\\workspace\\test_1\\train_x_y.csv', sep=' ', encoding='utf-8')
train_x = train.iloc[:,1:101]
train_y = train['y']
train_y=train_y.apply(c)
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.33, random_state=66)

"""
mod = joblib.load('ada.m')
pre_y = list(mod.predict(x_test))
y_a_score = mod.decision_function(x_test)
fpr_a, tpr_a, threshold_a = roc_curve(y_test, y_a_score)#fpr=FP/N tpr=TP/P
roc_auc_a = auc(fpr_a, tpr_a)
cm_m = confusion_matrix(y_test, pre_y)
print(cm_m)
print('正确率为：', accuracy_score(y_test, pre_y))
print('召回率为', recall_score(y_test, pre_y))

"""
clf1=LogisticRegression()
clf1.fit(x_train,y_train)
l_pre_y=list(clf1.predict(x_test))
y_l_score = clf1.decision_function(x_test)
fpr_l, tpr_l, threshold_l = roc_curve(y_test, y_l_score)
roc_auc_l = auc(fpr_l, tpr_l)
cm_l = confusion_matrix(y_test, l_pre_y)
print(cm_l)

print('l  正确率为：', accuracy_score(y_test, l_pre_y))
print('l  召回率为', recall_score(y_test, l_pre_y))
endtimel = datetime.datetime.now()
print(endtimel-starttime)


clf2 = svm.SVC(C=1)
clf2.fit(x_train,y_train)
s_pre_y=list(clf2.predict(x_test))
y_s_score = clf2.decision_function(x_test)
fpr_s, tpr_s, threshold_s = roc_curve(y_test, y_s_score)
roc_auc_s = auc(fpr_s, tpr_s)
cm_s = confusion_matrix(y_test, s_pre_y)
print(cm_s)
print('s  正确率为：',  accuracy_score(y_test, s_pre_y))
print('s  召回率为', recall_score(y_test, s_pre_y))
endtimes = datetime.datetime.now()
print(endtimes-starttime)




ada = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=6,min_samples_split=10), n_estimators=300, learning_rate=2)
ada.fit(x_train,y_train)
a_pre_y=list(ada.predict(x_test))

y_a2_score = ada.decision_function(x_test)
fpr_a2, tpr_a2, threshold_a2 = roc_curve(y_test, y_a2_score)
roc_auc_a2 = auc(fpr_a2, tpr_a2)
cm_a = confusion_matrix(y_test, a_pre_y)
print(cm_a)
print('a  正确率为：', accuracy_score(y_test, a_pre_y))
print('a  召回率为', recall_score(y_test, a_pre_y))
endtimea = datetime.datetime.now()
print(endtimea-starttime)

"""
ada1 = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=5), n_estimators=200, learning_rate=1)
ada1.fit(x_train,y_train)
a1_pre_y=list(ada1.predict(x_test))
print('a  正确率为：',  accuracy_score(y_test, a1_pre_y))
print('a  召回率为', recall_score(y_test, a1_pre_y))
endtimea = datetime.datetime.now()
print(endtimea-starttime)

"""


plt.figure()
plt.figure(figsize=(10, 10))

plt.plot(fpr_s, tpr_s, color='darkorange', lw=2, label='svm ROC curve (area = %0.2f)' % roc_auc_s)
plt.plot(fpr_l, tpr_l, color='navy', lw=2, label='LogisticRegression ROC curve (area = %0.2f)' % roc_auc_l)
plt.plot(fpr_a2, tpr_a2, color='red', lw=2, label='adaboost.改 ROC curve (area = %0.2f)' % roc_auc_a2)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

endtimea = datetime.datetime.now()
print(endtimea-starttime)

#confusion_matrix(y_test, y_pre)







