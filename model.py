# -*- coding: utf-8 -*-
# @Time    : 2019/4/22 15:07
# @Author  : Jason
# @FileName: model.py

import tensorflow as tf
import numpy as np


class Config(object):
    """
    配置参数
    """
    vec_dim = 128  # 输入的节点向量维度
    num_classes = 5
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    hidden_dim = 64  # 全连接神经元数目

    class_num = 5

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 0.1  # 学习率

    batch_size = 64  # 每批训练大小，即一个iterator训练64个样本，并且更新一次参数
    num_epochs = 500  # 总迭代次数

    print_per_batch = 10  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class GMCNN(object):
    def __init__(self, config):
        self.config = config
        # 待输入的数据
        # 输入的x的shape为[batch_size, vec_dim,vec_dim]
        self.input_x = tf.placeholder(tf.float32, [None, self.config.vec_dim*2], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # self.cnn()
        self.gmnn()

    # def cnn(self):
    #     """
    #     CNN模型
    #     :return:
    #     """
    #     with tf.name_scope("cnn"):
    #         # CNN layer
    #         conv1 = tf.layers.conv1d(self.input_x, 64, self.config.kernel_size, activation="relu", name="conv1")
    #         print("conv1: ", conv1.shape)
    #         # global max pooling layer
    #         gmp1 = tf.layers.max_pooling1d(conv1, pool_size=4, strides=2, name='gmp1')
    #         print("gmp1: ", gmp1.shape)
    #
    #         conv2 = tf.layers.conv1d(gmp1, 128, self.config.kernel_size, activation="relu", name="conv2")
    #         print("conv2: ", conv2.shape)
    #         gmp2 = tf.layers.max_pooling1d(conv2, pool_size=4, strides=2, name="gmp2")
    #         print("gmp2: ", gmp2.shape)
    #
    #         conv3 = tf.layers.conv1d(gmp2, self.config.num_filters, self.config.kernel_size, activation="relu",
    #                                  name="conv3")
    #         print("conv3: ", conv3.shape)
    #         gmp3 = tf.layers.max_pooling1d(conv3, pool_size=4, strides=2, name="gmp3")
    #         print("gmp3: ", gmp3.shape)
    #         gmp = tf.reduce_max(gmp3, reduction_indices=[1], name="gmp")
    #
    #     with tf.name_scope("score"):
    #         # 全连接层后面接dropout以及relu激活
    #         fc = tf.layers.dense(gmp, self.config.hidden_dim, name="fc1")
    #         fc = tf.contrib.layers.dropout(fc, self.config.dropout_keep_prob)
    #         fc = tf.nn.relu(fc)
    #         print("fc: ", fc.shape)
    #
    #         # 分类器
    #         self.logits = tf.layers.dense(fc, self.config.num_classes, name="fc2")
    #         self.y_pred_class = tf.argmax(tf.nn.softmax(self.logits), 1)  # softmax得到的是one-hot向量,取最大值对应的类别即为预测的类别
    #
    #     with tf.name_scope("optimize"):
    #         # 损失函数，交叉熵
    #         cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
    #         self.loss = tf.reduce_mean(cross_entropy)
    #         # 优化器
    #         self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
    #
    #     with tf.name_scope("accuracy"):
    #         # 准确率
    #         correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_class)
    #         self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    def gmnn(self):
        with tf.name_scope("concate"):
            con = tf.layers.dense(self.input_x, self.config.num_classes, name="con")
            con = tf.layers.dropout(con, self.config.dropout_keep_prob)
            con = tf.nn.relu(con)
        with tf.name_scope("score"):

            self.logits = tf.layers.dense(tf.nn.sigmoid(self.input_x), self.config.num_classes, name="fc")
            self.y_pred_class = tf.argmax(tf.nn.softmax(self.logits), 1)
        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_class)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
