import multiprocessing


from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd
file_in = 'np_f2.txt'
file_out = 'txt.model'
file_out2 = 'txt.vector'
model = Word2Vec(LineSentence(file_in), size=300, window=5, min_count=5, workers=multiprocessing.cpu_count())
model.save(file_out)
model.wv.save_word2vec_format(file_out2, binary=False)