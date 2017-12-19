# coding=utf-8

import os
import datetime
import numpy as np
import tensorflow as tf
from data_util import data_util_reuter
from text_cnn import text_cnn_model


# 参数设置
# 数据参数
tf.app.flags.DEFINE_string("positive_file_path", "../data/polarity.pos", "data source for the positive data.")
tf.app.flags.DEFINE_string("negative_file_path", "../data/polarity.neg", "data source for the negative data.")
# 模型参数
tf.app.flags.DEFINE_integer("sentence_size", 964, "sentence size (default: 60)")
tf.app.flags.DEFINE_integer("class_count", 8, "class count (default: 2)")
tf.app.flags.DEFINE_bool("is_use_exist_embedded", True, "whether to use embedded or not (default: True)")  # 是否使用已有词嵌入
tf.app.flags.DEFINE_integer("embedded_size", 128, "embedded word size (default: 128)")  # 词嵌入维度
tf.app.flags.DEFINE_string("filter_size_list", "3, 4, 5", "filter size list (default: 3, 4, 5)")
tf.app.flags.DEFINE_integer("filter_feature_count", 128, "filter feature count (default: 128)")
tf.app.flags.DEFINE_float("learning_rate", 0.1, "learning rate (default: 0.1)")  # 学习速率
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
    sentence_list_0, label_list_0 = data_util_reuter.loadSentenceAndLabel("../data/reuter/r8-train/acq.txt", "acq")
    sentence_list_1, label_list_1 = data_util_reuter.loadSentenceAndLabel("../data/reuter/r8-train/crude.txt", "crude")
    sentence_list_2, label_list_2 = data_util_reuter.loadSentenceAndLabel("../data/reuter/r8-train/earn.txt", "earn")
    sentence_list_3, label_list_3 = data_util_reuter.loadSentenceAndLabel("../data/reuter/r8-train/grain.txt", "grain")
    sentence_list_4, label_list_4 = data_util_reuter.loadSentenceAndLabel("../data/reuter/r8-train/interest.txt", "interest")
    sentence_list_5, label_list_5 = data_util_reuter.loadSentenceAndLabel("../data/reuter/r8-train/money-fx.txt", "money-fx")
    sentence_list_6, label_list_6 = data_util_reuter.loadSentenceAndLabel("../data/reuter/r8-train/ship.txt", "ship")
    sentence_list_7, label_list_7 = data_util_reuter.loadSentenceAndLabel("../data/reuter/r8-train/trade.txt", "trade")
    all_sentence_list = sentence_list_0 + sentence_list_1 + sentence_list_2 + sentence_list_3 + \
                        sentence_list_4 + sentence_list_5 + sentence_list_6 + sentence_list_7
    all_label_list = label_list_0 + label_list_1 + label_list_2 + label_list_3 + \
                     label_list_4 + label_list_5 + label_list_6 + label_list_7
    print("sentence count = %d" % len(all_sentence_list))
    # 2)build vocabulary
    _, vocabulary_size = data_util_reuter.createVocabulary(all_sentence_list)
    # 3)transform from sentence to data
    data_ndarray_0 = data_util_reuter.sentence2Data(sentence_list_0)
    data_ndarray_1 = data_util_reuter.sentence2Data(sentence_list_1)
    data_ndarray_2 = data_util_reuter.sentence2Data(sentence_list_2)
    data_ndarray_3 = data_util_reuter.sentence2Data(sentence_list_3)
    data_ndarray_4 = data_util_reuter.sentence2Data(sentence_list_4)
    data_ndarray_5 = data_util_reuter.sentence2Data(sentence_list_5)
    data_ndarray_6 = data_util_reuter.sentence2Data(sentence_list_6)
    data_ndarray_7 = data_util_reuter.sentence2Data(sentence_list_7)
    all_data_ndarray = np.concatenate((data_ndarray_0, data_ndarray_1, data_ndarray_2, data_ndarray_3,
                                       data_ndarray_4, data_ndarray_5, data_ndarray_6, data_ndarray_7), 0)
    # 4)create label dictionary
    label_dictionary = data_util_reuter.createLabelDictionary(all_label_list)
    print(label_dictionary)
    # 5)transform from label to data
    class_ndarray_0 = data_util_reuter.to_categorical(data_util_reuter.label2Class(label_list_0, label_dictionary), 8)
    print(class_ndarray_0[0])
    class_ndarray_1 = data_util_reuter.to_categorical(data_util_reuter.label2Class(label_list_1, label_dictionary), 8)
    print(class_ndarray_1[0])
    class_ndarray_2 = data_util_reuter.to_categorical(data_util_reuter.label2Class(label_list_2, label_dictionary), 8)
    print(class_ndarray_2[0])
    class_ndarray_3 = data_util_reuter.to_categorical(data_util_reuter.label2Class(label_list_3, label_dictionary), 8)
    print(class_ndarray_3[0])
    class_ndarray_4 = data_util_reuter.to_categorical(data_util_reuter.label2Class(label_list_4, label_dictionary), 8)
    print(class_ndarray_4[0])
    class_ndarray_5 = data_util_reuter.to_categorical(data_util_reuter.label2Class(label_list_5, label_dictionary), 8)
    print(class_ndarray_5[0])
    class_ndarray_6 = data_util_reuter.to_categorical(data_util_reuter.label2Class(label_list_6, label_dictionary), 8)
    print(class_ndarray_6[0])
    class_ndarray_7 = data_util_reuter.to_categorical(data_util_reuter.label2Class(label_list_7, label_dictionary), 8)
    print(class_ndarray_7[0])
    all_class_ndarray = np.concatenate((class_ndarray_0, class_ndarray_1, class_ndarray_2, class_ndarray_3,
                                        class_ndarray_4, class_ndarray_5, class_ndarray_6, class_ndarray_7), 0)
    # 6)shuffle data
    del all_sentence_list, all_label_list
    shuffle_data_ndarray, shuffle_class_ndarray = data_util_reuter.shuffleData(all_data_ndarray, all_class_ndarray)
    # 7)split
    train_x_ndarray, train_y_ndarray, valid_x_ndarray, valid_y_ndarray = \
        data_util_reuter.splitData(shuffle_data_ndarray, shuffle_class_ndarray, 0.1)
    del all_data_ndarray, all_class_ndarray, shuffle_data_ndarray, shuffle_class_ndarray
    # 8)train
    # (create session->train->validation，可选->prediction，可选)
    with tf.Graph().as_default():
        session_configure = tf.ConfigProto(allow_soft_placement=my_flag.allow_soft_placement,
                                           log_device_placement=my_flag.log_device_placement)
        session = tf.Session(config=session_configure)
        with session.as_default():
            # 定义模型
            model = text_cnn_model.TextCNN(sequence_size=my_flag.sentence_size,
                                           label_count=my_flag.class_count,
                                           vocabulary_size=vocabulary_size,
                                           embedded_size=my_flag.embedded_size,
                                           filter_size_list=list(map(int, my_flag.filter_size_list.split(", "))),
                                           filter_feature_count=my_flag.filter_feature_count,
                                           l2_regularization_lambda=my_flag.l2_regularization_lambda)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
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
                print("{}: step={}, loss={:g}, accuracy={:g}".format(time_str, step, loss, accuracy))

            # 验证函数
            def valid_step(input_x_batch, input_y_batch, writer=None):
                feed_dictionary = {model.input_x: input_x_batch, model.input_y: input_y_batch,
                                   model.dropout_keep_probability: 1.0}
                step, loss, accuracy = session.run(
                    [global_step, model.loss_function, model.accuracy],
                    feed_dictionary)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step={}, loss={:g}, accuracy={:g}".format(time_str, step, loss, accuracy))

            batch_list = data_util_reuter.generateBatch(list(zip(train_x_ndarray, train_y_ndarray)),
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