import pandas as pd
import numpy as np
from sklearn.externals import joblib
import  matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
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



mod = joblib.load('ada.m')
train = pd.read_csv('F:\\pycharm\\workspace\\test_1\\train_x_y.csv', sep=' ', encoding='utf-8')
train_x = train.iloc[:,1:101]
train_y = train['y']
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.33, random_state=66)

pre_y = list(mod.predict(x_test))
print('正确率为：', precision_score(y_test, pre_y))
print('召回率为', recall_score(y_test, pre_y))





