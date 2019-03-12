import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
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


def figure(np_f2):
    r = 0

    return r
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


if __name__ == '__main__':
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
    train_y = list(np_agg['label'])
    np_f = make(np_txt)  # 简单处理文本
    print('asdasdasdasd')
    #np_f2 = point(np_f)  # 提取关键词
    np_f2 = np_f
    print(16516161)
    with open("np_f2.txt", "w",encoding='utf-8') as f:
        for i in np_f2:
            f.write(i + '\n')

    print(11110101010)

    model = Word2Vec(LineSentence(file_in), size=300, window=5, min_count=0, workers=multiprocessing.cpu_count())
    model.save(file_out)
    model.wv.save_word2vec_format(file_out2, binary=False)
    model = gensim.models.KeyedVectors.load_word2vec_format('txt.vector', binary=False)
    output = filevec('np_f2.txt', model)
