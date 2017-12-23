# coding=utf-8

import re

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
    return string.strip().lower()  # 变小写，去两端空格

def readFile(file_path):
    with open(file_path, 'r', encoding='utf-8') as fin:
        sentence_list = fin.readlines()
        normalize_sentence_list = [normalizeString(sentence) for sentence in sentence_list]
        return normalize_sentence_list

def createSentimentDictionary(sentence_list):
    word_dictionary = {}
    for sentence in sentence_list:
        word_list = sentence.split(" ")
        for word in word_list:
            if word not in word_dictionary:
                word_dictionary[word] = 1
            else:
                word_dictionary[word] = word_dictionary[word] + 1
    return word_dictionary

sentence_list = readFile("../data/rt-polarity/rt-polarity-train/additional-negative-word.txt")
print(sentence_list)
word_dictionray = createSentimentDictionary(sentence_list)
print(word_dictionray)
