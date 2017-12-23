# coding=utf-8

import tensorflow as tf

class TextCNN(object):
    # 构造函数
    def __init__(self,
                 sequence_size, label_count,
                 vocabulary_size, embedded_size,
                 filter_size_list, filter_feature_count,
                 l2_regularization_lambda=0.0):
        # 输入层
        self.input_x = tf.placeholder(tf.int32, [None, sequence_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, label_count], name="input_y")
        # 嵌入层
        with tf.device("/:cpu:0"), tf.name_scope("embedded_layer"):
            self.embedded_w = tf.Variable(tf.random_uniform([vocabulary_size, embedded_size], -1.0, 1.0),
                                          name="embedded_w")
            embedded_out = tf.nn.embedding_lookup(self.embedded_w, self.input_x, name="embedded_out")  # 查找词向量
            embedded_out_expanded = tf.expand_dims(embedded_out, -1, name="embedded_out_expanded")  # 3维填充为4维
        # 卷积池化层
        pooled_out_list = []
        for i, filter_size in enumerate(filter_size_list):
            with tf.name_scope("convolutional_maxpool_layer_%s" % filter_size):
                # 卷积
                filter_shape = [filter_size, embedded_size, 1, filter_feature_count]
                convolutional_w = tf.Variable(tf.truncated_normal(filter_shape, 0.0, -0.1),
                                              name="convolutional_w")
                convolutional_b = tf.Variable(tf.constant(0.1, shape=[filter_feature_count]),
                                              name="convolutional_b")
                convolutional_out = tf.nn.conv2d(embedded_out_expanded, convolutional_w,
                                                       strides=[1, 1, 1, 1], padding="VALID",
                                                       name="convolutional_out")
                # 激活
                convolutional_h = tf.nn.relu(tf.nn.bias_add(convolutional_out, convolutional_b),
                                             name="convolutional_relu")
                # 池化
                pooled_out = tf.nn.max_pool(convolutional_h, ksize=[1, sequence_size-filter_size+1, 1, 1],
                                        strides=[1, 1, 1, 1], padding="VALID", name="pooled_out")
                pooled_out_list.append(pooled_out)
        filter_feature_total_count = filter_feature_count * len(filter_size_list)
        filter_feature_concat = tf.concat(pooled_out_list, 3)  # 串联池化结果
        filter_feature_flat = tf.reshape(filter_feature_concat, [-1, filter_feature_total_count])  # 展平
        # dropout
        with tf.name_scope("dropout"):
            self.dropout_keep_probability = tf.placeholder(tf.float32, name="dropout_keep_probability")
            dropout_out = tf.nn.dropout(filter_feature_flat, self.dropout_keep_probability, name="dropout_out")
        # 输出层
        with tf.name_scope("output_layer"):
            output_w = tf.get_variable("output_w", shape=[filter_feature_total_count, label_count],
                                       initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.1, shape=[label_count]), name="output_b")
            l2_loss = tf.constant(0.0)
            l2_loss = l2_loss + tf.nn.l2_loss(output_w) + tf.nn.l2_loss(output_b)
            self.output_out = tf.nn.xw_plus_b(dropout_out, output_w, output_b, name="output_out")
            output_label = tf.argmax(self.output_out, 1, name="output_label")
        # 损失函数
        with tf.name_scope("loss_function"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.output_out)
            self.loss_function = tf.reduce_mean(loss) + l2_regularization_lambda * l2_loss
        # 准确率
        with tf.name_scope("accuracy_result"):
            correct_prediction = tf.equal(output_label, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy_result")
