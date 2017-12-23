# coding=utf-8

import os
import re
import numpy as np
import tensorflow as tf
from tflearn.data_utils import VocabularyProcessor, to_categorical


# 规范化字符串。SST数据集不需要这么多规则，TREC数据集不需要转小写。
def normalizeString(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\‘]", " ", string)
    string = re.sub(r"\'s", " \'s", string)  # It's -> It 's
    string = re.sub(r"\'ve", " \'ve", string)  # I've -> I 've
    string = re.sub(r"n\'t", " n\'t", string)  # isn't -> is n't
    string = re.sub(r"\'re", " \'re", string)  # you're -> you 're
    string = re.sub(r"\'d", " \'d", string)  # you'd -> you 'd
    string = re.sub(r"\'ll", " \'ll", string)  # you'll -> you 'll
    string = re.sub(r",", " , ", string)  # , ->  ,
    string = re.sub(r"!", " ! ", string)  # ! ->  !
    string = re.sub(r"\(", " \( ", string)  # ( ->  (
    string = re.sub(r"\)", " \) ", string)  # ) ->  )
    string = re.sub(r"\?", " \? ", string)  # ? ->  ?
    string = re.sub(r"\s{2,}", " ", string)  # 2个以上空白格 -> 保留1个
    return string.strip().lower()

# 装载数据
def loadSentenceAndLabel(file_path, label):
    with open(file_path, "r", encoding="utf-8") as fin:
        sentence_list = fin.readlines()
        normalize_sentence_list = [normalizeString(sentence.strip()) for sentence in sentence_list]
        label_list = [label for _ in normalize_sentence_list]
    return normalize_sentence_list, label_list

# 创建词典
def createVocabulary(sentence_list, vocabulary_path=r"./vocabulary_reuter"):
    max_sentence_size = max([len(sentence.split(" ")) for sentence in sentence_list])
    my_vocabulary_processor = VocabularyProcessor(max_sentence_size)
    print("max sentence size = %d" % max_sentence_size)
    my_vocabulary_processor.fit(sentence_list)
    vocabulary_size = len(my_vocabulary_processor.vocabulary_)
    print("vocabulary size = %d" % vocabulary_size)
    my_vocabulary_processor.save(vocabulary_path)
    return my_vocabulary_processor, vocabulary_size

# 句子转数据
def sentence2Data(sentence_list, vocabulary_path=r"./vocabulary_reuter"):
    my_vocabulary_processor = VocabularyProcessor.restore(vocabulary_path)
    data_ndarray = np.array(list(my_vocabulary_processor.transform(sentence_list)))
    return data_ndarray

# 标签字典
def createLabelDictionary(label_list):
    my_label_dictionary = {}
    label_integer = 0
    for label in label_list:
        if label not in my_label_dictionary:
            my_label_dictionary[label] = label_integer
            label_integer += 1
        else:
            continue
    return my_label_dictionary

# 标签转数据
def label2Class(label_list, label_dictionary):
    label_integer_list = []
    for label in label_list:
        label_integer =label_dictionary[label]
        label_integer_list.append(label_integer)
    return np.array(list(label_integer_list))

# 打乱顺序
def shuffleData(data_ndarray, class_ndarray):
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(class_ndarray)))
    shuffle_data_ndarray = data_ndarray[shuffle_indices]
    shuffle_class_ndarray = class_ndarray[shuffle_indices]
    return shuffle_data_ndarray, shuffle_class_ndarray

# 分成训练集和验证集
def splitData(data_ndarry, class_ndarry, valid_percentage):
    valid_index = -1 * int(valid_percentage * float(len(data_ndarry)))
    train_x_ndarray, valid_x_ndarray = data_ndarry[:valid_index], data_ndarry[valid_index:]
    train_y_ndarray, valid_y_ndarray = class_ndarry[:valid_index], class_ndarry[valid_index:]
    return train_x_ndarray, train_y_ndarray, valid_x_ndarray, valid_y_ndarray

# 生成batch
def generateBatch(data_ndarray, batch_size, epoch_count, is_shuffle=True):
    data_size = len(data_ndarray)
    print(data_size)
    batch_count_per_epoch = int((len(data_ndarray)-1)/batch_size) + 1
    for epoch in range(epoch_count):
        # 是否全排列，得到一个新的列表
        if is_shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data_ndarray = data_ndarray[shuffle_indices]
        else:
            shuffled_data_ndarray = data_ndarray
        for batch_index in range(batch_count_per_epoch):
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, data_size)
            yield shuffled_data_ndarray[start_index: end_index]


if __name__ == "__main__":
    print("===")
    # 输出路径
    output_file_folder = os.path.abspath(os.path.join(os.path.curdir, "run"))
    # 1)load data
    sentence_list_0, label_list_0 = loadSentenceAndLabel("../data/reuter/r8-train/acq.txt", "acq")
    sentence_list_1, label_list_1 = loadSentenceAndLabel("../data/reuter/r8-train/crude.txt", "crude")
    sentence_list_2, label_list_2 = loadSentenceAndLabel("../data/reuter/r8-train/earn.txt", "earn")
    sentence_list_3, label_list_3 = loadSentenceAndLabel("../data/reuter/r8-train/grain.txt", "grain")
    sentence_list_4, label_list_4 = loadSentenceAndLabel("../data/reuter/r8-train/interest.txt", "interest")
    sentence_list_5, label_list_5 = loadSentenceAndLabel("../data/reuter/r8-train/money-fx.txt", "money-fx")
    sentence_list_6, label_list_6 = loadSentenceAndLabel("../data/reuter/r8-train/ship.txt", "ship")
    sentence_list_7, label_list_7 = loadSentenceAndLabel("../data/reuter/r8-train/trade.txt", "trade")
    all_sentence_list = sentence_list_0 + sentence_list_1 + sentence_list_2 + sentence_list_3 + \
                        sentence_list_4 + sentence_list_5 + sentence_list_6 + sentence_list_7
    all_label_list = label_list_0 + label_list_1 + label_list_2 + label_list_3 + \
                     label_list_4 + label_list_5 + label_list_6 + label_list_7
    print("sentence count = %d" % len(all_sentence_list))
    # 2)build vocabulary
    vocabulary_processor = createVocabulary(all_sentence_list)
    # 3)transform from sentence to data
    data_ndarray_0 = sentence2Data(sentence_list_0)
    data_ndarray_1 = sentence2Data(sentence_list_1)
    data_ndarray_2 = sentence2Data(sentence_list_2)
    data_ndarray_3 = sentence2Data(sentence_list_3)
    data_ndarray_4 = sentence2Data(sentence_list_4)
    data_ndarray_5 = sentence2Data(sentence_list_5)
    data_ndarray_6 = sentence2Data(sentence_list_6)
    data_ndarray_7 = sentence2Data(sentence_list_7)
    all_data_ndarray = np.concatenate((data_ndarray_0, data_ndarray_1, data_ndarray_2, data_ndarray_3,
                                       data_ndarray_4, data_ndarray_5, data_ndarray_6, data_ndarray_7), 0)
    # 4)create label dictionary
    label_dictionary = createLabelDictionary(all_label_list)
    print(label_dictionary)
    # 5)transform from label to data
    class_ndarray_0 = to_categorical(label2Class(label_list_0, label_dictionary), 8)
    print(class_ndarray_0[0])
    class_ndarray_1 = to_categorical(label2Class(label_list_1, label_dictionary), 8)
    print(class_ndarray_1[0])
    class_ndarray_2 = to_categorical(label2Class(label_list_2, label_dictionary), 8)
    print(class_ndarray_2[0])
    class_ndarray_3 = to_categorical(label2Class(label_list_3, label_dictionary), 8)
    print(class_ndarray_3[0])
    class_ndarray_4 = to_categorical(label2Class(label_list_4, label_dictionary), 8)
    print(class_ndarray_4[0])
    class_ndarray_5 = to_categorical(label2Class(label_list_5, label_dictionary), 8)
    print(class_ndarray_5[0])
    class_ndarray_6 = to_categorical(label2Class(label_list_6, label_dictionary), 8)
    print(class_ndarray_6[0])
    class_ndarray_7 = to_categorical(label2Class(label_list_7, label_dictionary), 8)
    print(class_ndarray_7[0])
    all_class_ndarray = np.concatenate((class_ndarray_0, class_ndarray_1, class_ndarray_2, class_ndarray_3,
                                        class_ndarray_4, class_ndarray_5, class_ndarray_6, class_ndarray_7), 0)
    # 6)shuffle data
    del sentence_list_0, sentence_list_1, sentence_list_2, sentence_list_3,\
        sentence_list_4, sentence_list_5, sentence_list_6, sentence_list_7, all_sentence_list,
    del label_list_0, label_list_1, label_list_2, label_list_3,\
        label_list_4, label_list_5, label_list_6, label_list_7, all_label_list
    shuffle_data_ndarray, shuffle_class_ndarray = shuffleData(all_data_ndarray, all_class_ndarray)
    # 7)split
    train_x_ndarry, train_y_ndarry, valid_x_ndarry, valid_y_ndarry = \
        splitData(shuffle_data_ndarray, shuffle_class_ndarray, 0.1)

