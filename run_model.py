# -*- coding: utf-8 -*-
# @Time    : 2019/4/24 16:49
# @Author  : Jason
# @FileName: run_model.py

import sys
from model import Config, GMCNN
import os
import tensorflow as tf
import time
from utils import preprocess, getUsersEdge, getMoviesEdge, batch_iter
from datetime import timedelta


def getTimeDif(time):
    return timedelta(seconds=int(round(time)))


def feed_data(x_batch, y_batch, dropout_keep_prob):
    # print("x_batch: ", x_batch.shape)
    # print("y_batch: ", y_batch.shape)
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: dropout_keep_prob
    }
    return feed_dict


def evaluate(session, x_val, y_val):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_val)
    batch_eval = batch_iter(x_val, y_val, model.config.batch_size)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, model.config.dropout_keep_prob)
        loss, acc = session.run([model.loss, model.accuracy], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置tensorboard
    tensorboard_dir = './tensorboard/GMCNN'
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar('accuracy', model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置saver
    saver_dir = './saver/GMCNN'
    saver = tf.train.Saver()
    if not os.path.exists(saver_dir):
        os.mkdir(saver_dir)

    print("Successfully configure tensorboard and saver.\n")

    print("Loading training and validation data...")
    user_vec_dir = './emb/users.emb'
    movie_vec_dir = './emb/movies.emb'
    if not os.path.exists(user_vec_dir):
        getUsersEdge(users_dir, user_vec_dir)
    if not os.path.exists(movie_vec_dir):
        getMoviesEdge(movies_dir, movie_vec_dir)

    # 载入训练集和验证集
    start_time = time.time()
    x_train, y_train = preprocess(user_vec_dir, movie_vec_dir, ratings_edgelist_dir, user_edge_dir, movie_edge_dir)
    x_val, y_val = preprocess(user_vec_dir, movie_vec_dir, ratings_val_edgelist_dir, user_edge_dir, movie_edge_dir)
    print("Training data x: ", x_train.shape)
    print("Training data x: ", x_train)
    print("Training data y: ", y_train.shape)
    print("Training data y: ", y_train)
    time_dif = time.time() - start_time
    print("Time usage: ", getTimeDif(time_dif))
    print("Successfully load training and evaluating data.\n")

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print("Training and evaluating...")
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 上一次准确率提升的批次
    require_improvement = 1000  # 超过1000次未提升，则提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print("Epoch :  ", epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.accuracy], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=saver_dir)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = getTimeDif(time.time() - start_time)
                msg = 'iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optimizer, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


# def test():
#     print("Loading test data...")
#     start_time = time.time()
#     x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)
#
#     session = tf.Session()
#     session.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     saver.restore(sess=session, save_path=save_path)  # 读取保存的模型
#
#     print('Testing...')
#     loss_test, acc_test = evaluate(session, x_test, y_test)
#     msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
#     print(msg.format(loss_test, acc_test))
#
#     batch_size = 128
#     data_len = len(x_test)
#     num_batch = int((data_len - 1) / batch_size) + 1
#
#     y_test_cls = np.argmax(y_test, 1)
#     y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
#     for i in range(num_batch):  # 逐批次处理
#         start_id = i * batch_size
#         end_id = min((i + 1) * batch_size, data_len)
#         feed_dict = {
#             model.input_x: x_test[start_id:end_id],
#             model.keep_prob: 1.0
#         }
#         y_pred_cls[start_id:end_id] = session.run(model.y_pred_class, feed_dict=feed_dict)
#
#     # 评估
#     print("Precision, Recall and F1-Score...")
#     print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))
#
#     # 混淆矩阵
#     print("Confusion Matrix...")
#     cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
#     print(cm)
#
#     time_dif = get_time_dif(start_time)
#     print("Time usage:", time_dif)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_model.py train/test""")
    users_dir = './data/users.txt'
    movies_dir = './data/movies.txt'
    ratings_edgelist_dir = './graph/ratings_small.edge'
    ratings_val_edgelist_dir = './graph/ratings_val.edge'
    user_edge_dir = "./graph/users.edge"
    movie_edge_dir = "./graph/movies.edge"
    print("Configuring GMCNN model...")
    config = Config()
    print("Successfully configure model.\n")
    model = GMCNN(config)
    if sys.argv[1] == 'train':
        train()
    # else:
    # test()
