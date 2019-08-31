import numpy as np
import re
import os
import calendar
import jieba
from typing import List

#######提取用户特征#######
def user_features():
    # train特征
    train_data_path = "dataset/original_dataset/weibo_train_data.txt"
    out_path = "dataset/train_user_features.txt"
    labels_out_path = "dataset/train_labels.txt"
    infile = open(train_data_path, "r", encoding="utf-8")
    lines = infile.readlines()
    infile.close()
    ZPZdict = {}
    user_index = 0
    ZPZ_slice = slice(3, 6)
    for line in lines:
        line_seg = line.split("\t")
        if line_seg[user_index] in ZPZdict:
            ZPZdict[line_seg[user_index]].append(list(map(int, line_seg[ZPZ_slice])))
        else:
            ZPZdict[line_seg[user_index]] = [list(map(int, line_seg[ZPZ_slice]))]

    temp = {}
    for k, v in ZPZdict.items():
        temp[k] = np.array(v)

    ZPZmeans = {}
    for k, v in temp.items():
        ZPZmeans[k] = np.sum(v, axis=0) / v.shape[0]

    res = []
    labels_res = []
    for line in lines:
        line_seg = line.split("\t")
        user = line_seg[user_index]
        labels_res.append(",".join(line_seg[3:6]) + "\n")
        res.append(",".join(map(str, ZPZmeans[user])) + "\n")

    outfile = open(out_path, "w", encoding="utf-8")
    outfile.writelines(res)
    outfile.close()

    outfile = open(labels_out_path, "w", encoding="utf-8")
    outfile.writelines(res)
    outfile.close()
    ###################################################################

    # predict特征
    test_data_path = "dataset/original_dataset/weibo_predict_data.txt"
    out_features_path = "dataset/predict_user_features.txt"
    out_uid_mid_path = "dataset/predict_uid_mid.txt"
    infile = open(test_data_path, "r", encoding="utf-8")
    lines = infile.readlines()
    infile.close()

    ZPZmeans_str = {}
    for k, v in ZPZmeans.items():
        ZPZmeans_str[k] = ",".join(map(str, v))
    res = []
    uid_mid = []
    for line in lines:
        line_seg = line.split("\t")
        uid_mid.append(line_seg[0] + "\t" + line_seg[1] + "\t\n")
        res.append(ZPZmeans_str.get(line_seg[user_index], "0.0,0.0,0.0") + "\n")

    outfile = open(out_features_path, "w", encoding="utf-8")
    outfile.writelines(res)
    outfile.close()

    outfile = open(out_uid_mid_path, "w", encoding="utf-8")
    outfile.writelines(uid_mid)
    outfile.close()

#######提取时间特征#######
def time_features():
    train_data_path = "dataset/original_dataset/weibo_train_data.txt"
    train_out_path = "dataset/train_time_features.txt"
    test_data_path = "dataset/original_dataset/weibo_predict_data.txt"
    test_out_path = "dataset/predict_time_features.txt"
    def handle(in_path, out_path):
        infile = open(in_path, "r", encoding="utf-8")
        lines = infile.readlines()
        infile.close()
        DATETIME_IN_LINE = 2
        DATE = 0
        TIME = 1
        res = []
        for line in lines:
            s = ""
            line_seg = line.split("\t")
            day = calendar.weekday(*map(int, line_seg[DATETIME_IN_LINE].split(" ")[DATE].split("-")))
            if day == 5 or day == 6:
                s += "0,"
            else:
                s += "1,"
            if '080000' < line_seg[DATETIME_IN_LINE].split(" ")[TIME].replace(":", "") < '180000':
                s += "1\n"
            else:
                s += "0\n"
            res.append(s)
        outfile = open(out_path, "w", encoding="utf-8")
        outfile.writelines(res)
        outfile.close()
    handle(train_data_path, train_out_path)
    handle(test_data_path, test_out_path)

#######提取文本特征#######
def text_features():
    train_data_path = "dataset/original_dataset/train_data_text.txt"
    train_out_path = "dataset/train_text_features.txt"
    test_data_path = "dataset/original_dataset/predict_data_text.txt"
    test_out_path = "dataset/predict_text_features.txt"
    def handle(in_path, out_path):
        pattern_topic = re.compile(r'#.*?#')
        pattern_title = re.compile(r'【.*?】')
        pattern_website = re.compile(r"((ht|f)tps?)://[\w\-]+(\.[\w\-]+)+([\w\-.,@?^=%&:/~+#]*[\w\-@?^=%&/~+#])?", re.ASCII)
        pattern_emoji = re.compile(r'[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u2B55]')
        pattern_expression = re.compile(r'\[.*?\]')

        pattern_non_words = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9]+')

        infile = open(in_path, "r", encoding="utf-8")
        lines = infile.readlines()
        infile.close()

        stopword_path = "dataset/original_dataset/stop_words.txt"
        infile = open(stopword_path, "r", encoding="utf-8")
        stopword = infile.readlines()
        stopword = set(map(str.strip, stopword))
        infile.close()


        def filter_non_words(s):
            if pattern_non_words.fullmatch(s) == None:
                return True
            else:
                return False

        def filter_numeric(s):
            if s.isnumeric():
                return False
            else:
                return True

        def filter_stopword(s):
            if s in stopword:
                return False
            else:
                return True

        def map_key(e):
            return e[0]

        def get_all_words(lines) -> List[List[str]]:
            filter_funcs = [filter_non_words, filter_numeric, filter_stopword]
            res = []
            for line in lines:
                fenci = jieba.cut(pattern_website.sub("", line))
                for func in filter_funcs:
                    fenci = filter(func, fenci)
                res.append(list(fenci))
            return res

        all_words = get_all_words(lines)

        def get_words_freq():
            words_freq = {}
            for line in all_words:
                for word in line:
                    if word in words_freq:
                        words_freq[word] += 1
                    else:
                        words_freq[word] = 1
            res = sorted(words_freq.items(), key=lambda e: e[1], reverse=True)
            res_files = []
            for item in res:
                res_files.append(item[0] + ":" + str(item[1]) + "\n")
            outfile = open("cipin.txt", "w", encoding="utf-8")
            outfile.writelines(res_files)
            outfile.close()
            return res

        # 主题 标题 链接 @ 表情 文本长度 高频词onehot
        ONE = "1,"
        ZERO = "0,"
        res = []
        high_freq_num = 40
        high_freq_words = get_words_freq()[:high_freq_num]
        h_words = list(map(map_key, high_freq_words))
        index = 0
        for line in lines:
            vec = ""
            if pattern_topic.search(line) != None:
                vec += ONE
            else:
                vec += ZERO
            if pattern_title.search(line) != None:
                vec += ONE
            else:
                vec += ZERO
            if pattern_website.search(line) != None:
                vec += ONE
            else:
                vec += ZERO
            if line.count("@") != 0:
                vec += ONE
            else:
                vec += ZERO
            if pattern_emoji.search(line) != None or pattern_expression.search(line) != None:
                vec += ONE
            else:
                vec += ZERO
            vec += (str(len(re.sub(r"\s+", "", line))) + ",")

            line_fenci = all_words[index]
            index += 1
            one_hot = ""
            for k in h_words:
                if k in line_fenci:
                    one_hot += "1"
                else:
                    one_hot += "0"
            vec += ",".join(one_hot)
            res.append(vec + "\n")

        outfile = open(out_path, "w", encoding="utf-8")
        outfile.writelines(res)
        outfile.close()
    
    handle(train_data_path, train_out_path)
    #handle(test_data_path, test_out_path)

    #合并文件
def merge_documents(user_features: object = False, time_features: object = False, text_features: object = False) -> object:
    all_train_path = [("dataset/train_user_features.txt", user_features),
                      ("dataset/train_time_features.txt", time_features),
                      ("dataset/train_text_features.txt", text_features)]
    train_data_path = (path[0] for path in all_train_path if path[1])

    all_predict_path = [("dataset/predict_user_features.txt", user_features),
                      ("dataset/predict_time_features.txt", time_features),
                      ("dataset/predict_text_features.txt", text_features)]
    predict_data_path = (path[0] for path in all_predict_path if path[1])

    if user_features or time_features or text_features:
        train_out_path = "dataset/train_features.txt"
        predict_out_path = "dataset/predict_features.txt"
    else:
        train_out_path = None
        predict_out_path = None

    def handle(in_path, out_path):
        lines_list = []
        for path in in_path:
            infile = open(path, "r", encoding="utf-8")
            lines_list.append(infile.readlines())
            infile.close()

        res = []
        if len(lines_list) != 0:
            for index in range(len(lines_list[0])):
                temp = ""
                for lines in lines_list:
                    temp += lines[index][:-1] + ","
                temp = temp[:-1] + "\n"
                res.append(temp)

        if out_path is not None:
            outfile = open(out_path, "w", encoding="utf-8")
            outfile.writelines(res)
            outfile.close()
    handle(train_data_path, train_out_path)
    handle(predict_data_path, predict_out_path)

    infile = open("dataset/original_dataset/weibo_train_data.txt", "r", encoding="utf-8")
    lines = infile.readlines()
    infile.close()

    # 提取train labels

    

if __name__ == '__main__':

    # user_features()
    # time_features()
    text_features()
    # merge_documents(user_features=True, time_features=True)

