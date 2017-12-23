# coding=utf-8

import os
import datetime
import numpy as np
import tensorflow as tf
from data_util import data_util_mr
from text_cnn_mr import text_cnn_mr_model


# 参数设置
# 数据参数
# tf.app.flags.DEFINE_string("positive_file_path", "../data/polarity.pos", "data source for the positive data.")
# tf.app.flags.DEFINE_string("negative_file_path", "../data/polarity.neg", "data source for the negative data.")
# 模型参数
tf.app.flags.DEFINE_integer("sentence_size", 56, "sentence size (default: 60)")
tf.app.flags.DEFINE_integer("class_count", 2, "class count (default: 2)")
tf.app.flags.DEFINE_bool("is_use_exist_embedded", True, "whether to use embedded or not (default: True)")  # 是否使用已有词嵌入
tf.app.flags.DEFINE_integer("embedded_size", 128, "embedded word size (default: 128)")  # 词嵌入维度
tf.app.flags.DEFINE_string("filter_size_list", "3, 4, 5", "filter size list (default: 3, 4, 5)")
tf.app.flags.DEFINE_integer("filter_feature_count", 128, "filter feature count (default: 128)")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "learning rate (default: 0.1)")  # 学习速率
tf.app.flags.DEFINE_float("dropout_keep_probability", 0.5, "dropout keep probability (default: 0.5)")
tf.app.flags.DEFINE_float("l2_regularization_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
# 过程参数
tf.app.flags.DEFINE_float("valid_data_percentage", 0.1, "percentage of the data to use for valid (default: 0.1)")
tf.app.flags.DEFINE_integer("epoch_count", 200, "count of train epochs (default: 200)")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size (default:64)")
tf.app.flags.DEFINE_integer("keep_checkpoint_count", 5, "count of checkpoints (default: 5)")
tf.app.flags.DEFINE_integer("checkpoint_step_every", 100, "save model after epoch (default: 10)")
tf.app.flags.DEFINE_integer("evaluate_step_every", 100, "evaluate model on deviation set after steps (default: 100)")
# 其他参数
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")  # 允许软切换运算设备
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of operations on devices")  # 打印运算设备信息
# 显示所有参数
my_flag = tf.app.flags.FLAGS  # 创建一个类
my_flag._parse_flags()  # 可以把__flags变成一个字典
print("\nParameters:")
for attribute, value in sorted(my_flag.__flags.items()):
    print("{0:25} = {1}".format(attribute.upper(), value))
print("")


# 1)load data->2)build vocabulary->3)split data->4)train
def main(_):
    # 输出路径
    output_file_folder = os.path.abspath(os.path.join(os.path.curdir, "run"))
    # 1)load data
    sentence_list_0, label_list_0 = data_util_mr.loadSentenceAndLabel\
        ("../data/rt-polarity/rt-polarity-train/rt-polarity.neg", "negative")
    sentence_list_1, label_list_1 = data_util_mr.loadSentenceAndLabel\
        ("../data/rt-polarity/rt-polarity-train/rt-polarity.pos", "positive")
    sentence_list_a0, label_list_a0 = data_util_mr.loadSentenceAndLabel \
        ("../data/rt-polarity/rt-polarity-train/additional-negative-word.txt", "negative")
    sentence_list_a1, label_list_a1 = data_util_mr.loadSentenceAndLabel \
        ("../data/rt-polarity/rt-polarity-train/additional-positive-word.txt", "positive")
    all_sentence_list = sentence_list_0 + sentence_list_1 + sentence_list_a0 + sentence_list_a1
    all_label_list = label_list_0 + label_list_1 + label_list_a0 + label_list_a1
    print("sentence count = %d" % len(all_sentence_list))
    # 2)build vocabulary
    _, vocabulary_size = data_util_mr.createVocabulary(all_sentence_list)
    # 3)transform from sentence to data
    data_ndarray_0 = data_util_mr.sentence2Data(sentence_list_0)
    data_ndarray_1 = data_util_mr.sentence2Data(sentence_list_1)
    data_ndarray_a0 = data_util_mr.sentence2Data(sentence_list_a0)
    data_ndarray_a1 = data_util_mr.sentence2Data(sentence_list_a1)
    all_data_ndarray = np.concatenate((data_ndarray_0, data_ndarray_1), 0)
    # 4)create label dictionary
    label_dictionary = data_util_mr.createLabelDictionary(all_label_list)
    print(label_dictionary)
    # 5)transform from label to data
    class_ndarray_0 = data_util_mr.to_categorical(data_util_mr.label2Class(label_list_0, label_dictionary), 2)
    print(class_ndarray_0[0])
    class_ndarray_1 = data_util_mr.to_categorical(data_util_mr.label2Class(label_list_1, label_dictionary), 2)
    print(class_ndarray_1[0])
    class_ndarray_a0 = data_util_mr.to_categorical(data_util_mr.label2Class(label_list_a0, label_dictionary), 2)
    print(class_ndarray_a0[0])
    class_ndarray_a1 = data_util_mr.to_categorical(data_util_mr.label2Class(label_list_a1, label_dictionary), 2)
    print(class_ndarray_a1[0])
    all_class_ndarray = np.concatenate((class_ndarray_0, class_ndarray_1), 0)
    # 6)shuffle data
    del sentence_list_0, sentence_list_1, all_sentence_list
    del label_list_0, label_list_1, all_label_list
    shuffle_data_ndarray, shuffle_class_ndarray = data_util_mr.shuffleData(all_data_ndarray, all_class_ndarray)
    # 7)split
    train_x_ndarray, train_y_ndarray, valid_x_ndarray, valid_y_ndarray = \
        data_util_mr.splitData(shuffle_data_ndarray, shuffle_class_ndarray, my_flag.valid_data_percentage)
    train_x_ndarray = np.concatenate((train_x_ndarray, data_ndarray_a0, data_ndarray_a1), 0)
    train_y_ndarray = np.concatenate((train_y_ndarray, class_ndarray_a0, class_ndarray_a1), 0)

    del all_data_ndarray, all_class_ndarray, shuffle_data_ndarray, shuffle_class_ndarray
    # 8)train
    # (create session->train->validation，可选->prediction，可选)
    with tf.Graph().as_default():
        session_configure = tf.ConfigProto(allow_soft_placement=my_flag.allow_soft_placement,
                                           log_device_placement=my_flag.log_device_placement)
        session = tf.Session(config=session_configure)
        with session.as_default():
            # 定义模型
            model = text_cnn_mr_model.TextCNN(sequence_size=my_flag.sentence_size,
                                                  label_count=my_flag.class_count,
                                                  vocabulary_size=vocabulary_size,
                                                  embedded_size=my_flag.embedded_size,
                                                  filter_size_list=list(map(int, my_flag.filter_size_list.split(", "))),
                                                  filter_feature_count=my_flag.filter_feature_count,
                                                  l2_regularization_lambda=my_flag.l2_regularization_lambda)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(my_flag.learning_rate)
            grads_and_vars = optimizer.compute_gradients(model.loss_function)
            train_operation = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            # 初始化变量
            init_operation = session.run(tf.global_variables_initializer())

            # 训练函数
            def train_step(input_x_batch, input_y_batch):
                feed_dictionary = {model.input_x: input_x_batch, model.input_y: input_y_batch,
                                   model.dropout_keep_probability: my_flag.dropout_keep_probability}
                _, step, loss, accuracy = session.run(
                    [train_operation, global_step, model.loss_function, model.accuracy],
                    feed_dictionary)
                time_str = datetime.datetime.now().isoformat()
                # print("{}: step ={:10d}, loss ={:10.7g}, accuracy ={:10.7g}".format(time_str, step, loss, accuracy))

            # 验证函数
            def valid_step(input_x_batch, input_y_batch):
                feed_dictionary = {model.input_x: input_x_batch, model.input_y: input_y_batch,
                                   model.dropout_keep_probability: 1.0}
                step, loss, accuracy = session.run(
                    [global_step, model.loss_function, model.accuracy],
                    feed_dictionary)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step ={:10d}, loss ={:10.7g}, accuracy ={:10.7g}".format(time_str, step, loss, accuracy))

            batch_list = data_util_mr.generateBatch(list(zip(train_x_ndarray, train_y_ndarray)),
                                                        my_flag.batch_size,
                                                        my_flag.epoch_count,
                                                        False)
            for batch in batch_list:
                batch_x, batch_y = zip(*batch)
                train_step(batch_x, batch_y)
                current_step = tf.train.global_step(session, global_step)
                if current_step % my_flag.evaluate_step_every == 0:
                    print("\nEvaluation:")
                    valid_step(valid_x_ndarray, valid_y_ndarray)
                    print("")


if __name__ == "__main__":
    tf.app.run()
