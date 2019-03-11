import pandas as pd
import jieba
import jieba.analyse
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
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


if __name__ == '__main__':
    stopwords = stopwordslist('stop.txt')  # 加载停用词
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
    np_f2 = point(np_f)  # 提取关键词
    with open("np_f2.txt", "w") as f:
        for i in np_f2:
            f.write(i + '\n')
    file_in = 'np_f2.txt'
    file_out = 'txt.model'
    file_out2 = 'txt.vector'
    model = Word2Vec(LineSentence(file_in), size=300, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.save(file_out)
    model.wv.save_word2vec_format(file_out2, binary=False)