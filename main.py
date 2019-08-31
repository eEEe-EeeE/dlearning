import pandas as pd
import numpy as np
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts
from gensim.models.word2vec import LineSentence
from gensim.test.utils import get_tmpfile


def main():
    import chardet
    delimite = 614809
    num = 0
    f1 = open("weibo/dataset/weibo_train_data1.txt", "r", encoding="utf-8")
    f2 = open("weibo/dataset/weibo_train_data2.txt", "r", encoding="utf-8")
    print(len(f1.readlines()) + len(f2.readlines()) == delimite * 2)
    


if __name__ == '__main__':
    main()
