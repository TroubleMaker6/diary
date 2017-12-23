# coding=utf-8

import datetime
import numpy as np
import tensorflow as tf
from data_util import data_util_enwik8
from word2vec import skip_gram_text8_model


# 参数设置
# 模型参数
tf.app.flags.DEFINE_integer("word_count", 128, "word count per batch (default:128)")
tf.app.flags.DEFINE_integer("label_count", 128, "label count per batch (default:128)")
tf.app.flags.DEFINE_integer("vocabulary_size", 50000, "vocabulary size (default:50000)")
tf.app.flags.DEFINE_integer("embedded_size", 128, "embedded word size (default:128)")
tf.app.flags.DEFINE_float("learning_rate", 1.0, "learning rate (default:1.0)")
# 过程参数
tf.app.flags.DEFINE_float("valid_percentage", 0.1, "percentage for valid (dafault:0.1)")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch size (default:128)")
tf.app.flags.DEFINE_integer("epoch_count", 100000, "epoch count (default:100000)")
tf.app.flags.DEFINE_integer("train_step_every", 1000, "train every step (default:1000)")
tf.app.flags.DEFINE_integer("evaluate_step_every", 10000, "evaluate every step (default: 10000)")
# 其他参数
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "all device sotf device placement (default:True)")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "log placement of operation on device (default:False)")
# 显示所有参数
my_flag = tf.app.flags.FLAGS
my_flag._parse_flags()
print("\nParameters:")
for attribute, value in sorted(my_flag.__flags.items()):
    print("{0:25} = {1}".format(attribute.upper(), value))
print("")

# 1)load data->2)build vocabulary->3)split data->4)train
def main(_):
    # 1)load data
    file_path = data_util_enwik8.maybeDownload("text8.zip", 31344016)
    word_list = data_util_enwik8.readZipFile(file_path)
    print("word count = %d" % (len(word_list)))
    # 2)build vocabulary
    data_ndarray, word_count_list, word_index_dictionary, index_word_dictionary = data_util_enwik8.createVocabulary(word_list)
    print("most common word (+UNK):", word_count_list[:5])
    print("word list:", [index_word_dictionary[i] for i in data_ndarray[:20]])
    print("data list:", data_ndarray[:20])
    del word_list
    # 3)split data
    valid_data = np.random.choice(100, 16, replace=False)
    # 4)train
    # create session->train->validation->prediction
    with tf.Graph().as_default():
        session_configure = tf.ConfigProto(allow_soft_placement=my_flag.allow_soft_placement,
                                           log_device_placement=my_flag.log_device_placement)
        session = tf.Session(config=session_configure)
        with session.as_default():
            # 定义网络
            model = skip_gram_text8_model.SkipGram(word_batch_size=my_flag.batch_size,
                                                   label_batch_size=my_flag.batch_size,
                                                   vocabulary_size=my_flag.vocabulary_size,
                                                   embedded_size=my_flag.embedded_size)
            # 定义步数变量
            global_step = tf.Variable(0, trainable=False, name="global_step")
            # 定义优化器
            optimizer = tf.train.GradientDescentOptimizer(my_flag.learning_rate)
            # 定义训练操作
            grads_and_vars = optimizer.compute_gradients(model.loss_function)
            train_operation = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # 训练函数
            def trainStep(batch_input_x, batch_input_y):
                feed_dictionary = {model.input_x: batch_input_x, model.input_y:batch_input_y}
                _, step, loss = session.run(
                    [train_operation, global_step, model.loss_function],
                    feed_dict=feed_dictionary
                )
                time_str = datetime.datetime.now().isoformat()
                print("{} : step = {:10d}, loss = {:10.7g}".format(time_str, step, loss))
                return loss

            # 验证函数
            def validStep():
                # 验证效果
                valid_window = 100
                valid_size = 10
                valid_data = np.random.choice(valid_window, valid_size, replace=False)
                valid_x = tf.constant(valid_data, dtype=tf.int32)
                valid_embedded_data = tf.nn.embedding_lookup(model.normalized_embedded_w, valid_x)
                similarity_function = tf.matmul(valid_embedded_data, model.normalized_embedded_w, transpose_b=True)
                similarity_rate = session.run(similarity_function)
                for i in range(valid_size):
                    valid_word = index_word_dictionary[valid_data[i]]
                    top_k = 8
                    nearest_data = (-similarity_rate[i, :]).argsort()[1:top_k+1]
                    log_str = "nearest word to %s:" % valid_word
                    for k in range(top_k):
                        close_word = index_word_dictionary[nearest_data[k]]
                        log_str = "%s %s," %(log_str, close_word)
                    print(log_str)

            # 开始训练
            # 初始化变量
            init_operation = session.run(tf.global_variables_initializer())
            average_loss = 0
            for i in range(my_flag.epoch_count):
                batch_word, batch_label = data_util_enwik8.generateBatch(data_ndarray, my_flag.batch_size, 2, 1)
                loss = trainStep(batch_word, batch_label)
                # 展示训练过程
                current_step = tf.train.global_step(session, global_step)
                average_loss += loss
                if current_step % my_flag.train_step_every == 0:
                    if current_step > 0:
                        average_loss /= my_flag.train_step_every
                    time_str = datetime.datetime.now().isoformat()
                    print("\nTrain:")
                    print("{} : step = {:10d}, avg_loss = {:10.7g}".format(time_str, current_step, average_loss))
                    print("")
                    average_loss = 0
                if current_step % my_flag.evaluate_step_every == 0:
                    print("\nEvaluation:")
                    validStep()
                    print("")


if __name__ == "__main__":
    tf.app.run()

    # final_embedding_w = session.run(model.normalized_embedded_w)