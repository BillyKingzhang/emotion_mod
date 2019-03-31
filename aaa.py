import pandas as pd
import numpy as np
from sklearn.externals import joblib
import  matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
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

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]
    return stopwords


def make(np_txt):
    rs_rs = []
    rs_f = []
    np_rs = np_txt
    """
    for list in np_txt:
        si = list.find("：")
        np_rs.append(list[si:])
    """
    for i in range(len(np_rs)):
        rs = jieba.cut("".join(np_rs[i].split()))
        rs1 = " ".join(rs)
        rs1 = rs1.split(" ")
        for w in rs1:
            if w not in stopwords:
                rs_rs.append(w)
        rs_f.append(" ".join(rs_rs))
        rs_rs = []
    return rs_f  ###3


def point(np_f):
    key = []
    out = []
    for i in np_f:
        keywords = jieba.analyse.extract_tags(i, topK=10, withWeight=True, allowPOS=())
        for item in keywords:
            # 分别为关键词和相应的权重
            key.append(item[0])
        out.append(" ".join(key))
        key = []
    return out


def wordvec(wlist, model):
        list = []
        for i in wlist:
            i = i.replace('\n', '')
            if i not in model:
                list.append(np.zeros(300))
                continue
            list.append(model[i])

        return np.array(list, dtype='float')


def filevec(filename, model):
        fvec = []
        with open(filename, 'r',encoding='utf-8') as f:
            for line in f:
                wordlist = line.split(' ')
                vecs = wordvec(wordlist, model)
                if len(vecs) > 0:
                    vecarray = sum(np.array(vecs)) / len(vecs)
                    fvec.append(vecarray)
        return fvec


def c(x):
    if x==0:
        return 0
    else:
        return 1


if __name__ == '__main__':

    starttime = datetime.datetime.now()
    stopwords = stopwordslist('stop_words.txt')  # 加载停用词
    """
   
    negwords = stopwordslist('neg.txt')  # 加载正面词
    poswords = stopwordslist('pos.txt')  # 加载负面词
    file_name = "yao_shen.csv"
    np_agg = pd.read_csv(file_name, encoding='utf-8', sep=',')  # 加载文本
    np_txt = list(np_agg['WB_text'])
    print(121212121212121)

    np_f = make(np_txt)  # 简单处理文本

    np_f2 = point(np_f)  # 提取关键词

    result = figure(np_f2)  # 计算感情倾向

    print("此文本感情倾向值为:", result)
    """

    file_in = 'np_f2.txt'
    file_out = 'txt.model'
    file_out2 = 'txt.vector'
    file_name = 'simplifyweibo_4_moods.csv'
    np_agg = pd.read_csv(file_name, encoding='utf-8', sep=',')  # 加载文本
    np_txt = list(np_agg['review'])
    train_y = list(np_agg['label'].apply(c))
    print(00000000)
    np_f = make(np_txt)  # 简单处理文本

    #np_f2 = point(np_f)  # 提取关键词
    np_f2 = np_f

    with open("np_f2.txt", "w",encoding='utf-8') as f:
        for i in np_f2:
            f.write(i + '\n')

    model = Word2Vec(LineSentence(file_in), size=300, window=5, min_count=0, workers=multiprocessing.cpu_count())
    model.save(file_out)
    model.wv.save_word2vec_format(file_out2, binary=False)
    model = gensim.models.KeyedVectors.load_word2vec_format('txt.vector', binary=False)
    output = filevec('np_f2.txt', model)
    trianx = pd.DataFrame(output)
    print(111111111111)
    trianx['y'] = train_y
    trianx.to_csv('train_x_y.csv', sep=' ', encoding='utf-8')
    t_x = trianx
    del t_x['y']
    """
    pac =PCA(n_components =300)
    pac.fit(t_x)
    plt.figure(1,figsize=(4,3))
    plt.clf()
    plt.axes([.2,.2,.7,.7])
    plt.plot(pac.explained_variance_,linewidth=2)
    plt.xlabel('tight')
    plt.ylabel('ex_v')
    plt.show()
    """
    train_x = t_x.iloc[:,0:100]
    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size = 0.33, random_state = 66)
    ada = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=5), n_estimators=200, learning_rate=1)
    """
    clf = svm.SVC(C=2) 
    clf.fit(x_train,y_train)
    """
    ada.fit(x_train, y_train)
    joblib.dump(ada, 'ada.m')
    print('test:', ada.score(x_test, y_test))
    print('trian:', ada.score(x_train, y_train))
    endtime = datetime.datetime.now()
    print(endtime - starttime)





