# coding=utf-8

import os
import zipfile
import collections
import random
import numpy as np
import tensorflow as tf
from urllib.request import urlretrieve


url = "http://mattmahoney.net/dc/"

def maybeDownload(file_name, expected_file_size):
    download_file_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), r"data\enwik8\train")
    download_file_path = os.path.join(download_file_folder, file_name)
    if not os.path.exists(download_file_path):
        download_file_path, _ = urlretrieve(url+file_name, download_file_path)
    file_state_information = os.stat(download_file_path)
    if file_state_information.st_size == expected_file_size:
        print("download corpus complete")
    else:
        print("download corpus failed")
    return download_file_path

def readZipFile(file_path):
    with zipfile.ZipFile(file_path) as fin:
        word_list = tf.compat.as_str(fin.read(fin.namelist()[0])).split()
        return word_list

def readTxtFile(file_path):
    with open(file_path) as fin:
        word_list =  tf.compat.as_str(fin.read()).split()
        return word_list

def createVocabulary(word_list):
    vocabulary_size = 50000
    word_count_list = [["UNK", -1]]
    word_count_list.extend(collections.Counter(word_list).most_common(vocabulary_size-1))
    word_index_dictionary = dict()
    for word, _ in word_count_list:
        word_index_dictionary[word] = len(word_index_dictionary)
    data_list = list()
    unknown_count = 0
    for word in word_list:
        word_index = word_index_dictionary.get(word, 0)
        if word_index == 0:
            unknown_count += 1
        data_list.append(word_index)
    word_count_list[0][1] = unknown_count
    index_word_dictionary = dict(zip(word_index_dictionary.values(), word_index_dictionary.keys()))
    return data_list, word_count_list, word_index_dictionary, index_word_dictionary

data_position = 0
def generateBatch(data_list, feature_count_per_batch, feature_count_per_word, oneside_feature_window_per_word):
    data_ndarray = np.array(data_list)
    global data_position
    assert feature_count_per_batch % feature_count_per_word == 0  # 一共产生整数词的样本
    word_count_per_batch = feature_count_per_batch // feature_count_per_word
    assert feature_count_per_word <= 2*oneside_feature_window_per_word  # 保证能能产生足够样本
    batch_data = np.ndarray(shape=(feature_count_per_batch,), dtype=np.int32)
    batch_label = np.ndarray(shape=(feature_count_per_batch, 1), dtype=np.int32)
    span = 2 * oneside_feature_window_per_word + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data_ndarray[data_position])
        data_position = (data_position+1) % len(data_ndarray)
    for i in range(word_count_per_batch):
        target_position = oneside_feature_window_per_word
        target_to_avoid_position = [oneside_feature_window_per_word]
        for j in range(feature_count_per_word):
            while target_position in target_to_avoid_position:
                target_position = random.randint(0, span-1)
            target_to_avoid_position.append(target_position)
            batch_data[i*feature_count_per_word+j] = buffer[oneside_feature_window_per_word]
            batch_label[i*feature_count_per_word+j, 0] = buffer[target_position]
        buffer.append(data_ndarray[data_position])
        data_position = (data_position + 1) % len(data_ndarray)
    return batch_data, batch_label


if __name__ == "__main__":
    file_path = maybeDownload("text8.zip", 31344016)
    word_list = readZipFile(file_path)
    print("word count = %d" % (len(word_list)))
    data_ndarray, word_count_list, word_index_dictionary, index_word_dictionary = createVocabulary(word_list)
    print(word_list[:10])
    del word_list
    print("most common word (+UNK):", word_count_list[:5])
    print("word list:", [index_word_dictionary[i] for i in data_ndarray[:20]])
    print("data list:", data_ndarray[:20])
    for i in range(2):
        batch, label = generateBatch(data_ndarray, 8, 2, 1)
        print(batch)
        print(label.reshape((1, -1)))
