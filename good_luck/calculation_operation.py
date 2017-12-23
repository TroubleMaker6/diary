# coding=utf-8

import tensorflow as tf


a = tf.constant([1, 2, 3], dtype=tf.int32)
b = tf.reduce_sum(tf.square(a))

with tf.Session() as sess:
    c = sess.run(b)
    print(c)