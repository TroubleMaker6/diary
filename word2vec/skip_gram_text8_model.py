# coding=utf-8

import math
import numpy as np
import tensorflow as tf


class SkipGram(object):
    # 构造函数
    def __init__(self,
                 word_batch_size, label_batch_size,
                 vocabulary_size, embedded_size):
        # 输入层
        self.input_x = tf.placeholder(tf.int32, shape=(word_batch_size), name="input_x")
        self.input_y = tf.placeholder(tf.int32, shape=(label_batch_size, 1), name="input_y")
        # 嵌入层
        with tf.device("/:cpu:0"), tf.name_scope("embedded_layer"):
            self.embedded_w = tf.Variable(tf.random_uniform(shape=(vocabulary_size, embedded_size),
                                                            minval=-1.0, maxval=1.0), name="embedded_w")
            embedded_out = tf.nn.embedding_lookup(self.embedded_w, self.input_x, name="embedded_out")
        # 损失函数
        with tf.name_scope("loss_function"):
            nce_w = tf.Variable(tf.truncated_normal([vocabulary_size, embedded_size],
                                                    stddev=1.0/math.sqrt(embedded_size)),
                                name="nce_w")
            nce_b = tf.Variable(tf.zeros([vocabulary_size]), name="nce_b")
            self.loss_function = tf.reduce_mean(tf.nn.nce_loss(weights=nce_w,
                                                               biases=nce_b,
                                                               labels=self.input_y,
                                                               inputs=embedded_out,
                                                               num_sampled=64,
                                                               num_classes=vocabulary_size))
        # 归一化词向量
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedded_w), 1, keep_dims=True))
        self.normalized_embedded_w = tf.div(self.embedded_w, norm, name="normalized_embedded_w")
