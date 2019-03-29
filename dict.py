import multiprocessing
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd




from sklearn.model_selection import KFold

file_in = 'np_f2.txt'
file_out = 'txt.model'
file_out2 = 'txt.vector'


def wordvec(wlist, model):
    list = []
    for i in wlist:
        i = i.replace('\n', '')
        list.append(model[i])

    return np.array(list, dtype='float')


def filevec(filename, model):
    fvec = []
    with open(filename, 'r') as f:
        for line in f:
            wordlist = line.split(' ')
            vecs = wordvec(wordlist, model)
            if len(vecs) > 0:
                vecarray = sum(np.array(vecs)) / len(vecs)
                fvec.append(vecarray)
    return fvec


model = Word2Vec(LineSentence(file_in), size=300, window=0, min_count=5, workers=multiprocessing.cpu_count())
model.save(file_out)
model.wv.save_word2vec_format(file_out2, binary=False)
model = gensim.models.KeyedVectors.load_word2vec_format('txt.vector', binary=False)
output = filevec('np_f2.txt', model)


kf = KFold(n_splits=2)


print(kf)

for train_index, test_index in kf.split(x):  #几折循环几次，相应得到几次结果，这里参数写Y也一样
    print('TRAIN:', train_index, "TEST:",  kf.split(x))
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]











