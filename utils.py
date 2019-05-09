# -*- coding: utf-8 -*-
# @Time    : 2019/4/22 16:47
# @Author  : Jason
# @FileName: utils.py
import numpy as np


def getUsersEdge(user_dir, user_egdelist_dir):
    userID, genders, ages, occupations = [], [], [], []
    with open(user_dir, 'r', encoding='utf-8') as lines:
        for line in lines:
            userID.append(line.strip().split("::")[0])
            genders.append(line.strip().split("::")[1])
            ages.append(line.strip().split("::")[2])
            occupations.append(line.strip().split("::")[3])
    lines.close()
    with open(user_egdelist_dir, 'w', encoding='utf-8') as user_writer:
        for i in range(len(userID)):
            for j in range(len(userID)):
                if i != j and genders[i] == genders[j] and ages[i] == ages[j] and occupations[i] == occupations[j]:
                    user_writer.write(str(userID[i]) + " " + str(userID[j]) + "\n")
    user_writer.close()


def cmptype(type1, type2):
    for i in type1:
        if i in type2:
            return True
    return False


def getMoviesEdge(movie_dir, movie_edgelist_dir):
    movieID, types = [], []
    with open(movie_dir, 'r', encoding='utf-8') as lines:
        for line in lines:
            movieID.append(line.strip().split("::")[0])
            types.append(line.strip().split("::")[-1].split("|"))
    lines.close()
    with open(movie_edgelist_dir, 'w', encoding='utf-8') as movie_writer:
        for i in range(len(movieID)):
            for j in range(len(movieID)):
                if i != j and types[i] == types[j]:
                    movie_writer.write(str(movieID[i]) + " " + str(movieID[j]) + "\n")
    movie_writer.close()


def getRatingEdge(user_vec_dir, movie_vec_dir, ratings_dir, training_data_dir):
    flag = False
    user_dic = {}
    with open(user_vec_dir, 'r', encoding='utf-8') as lines:
        for line in lines:
            if flag:
                user_node = line.split(" ")[0]
                user_node_vec = list(map(float, line.split(" ")[1:]))
                if user_node not in user_dic:
                    user_dic[user_node] = user_node_vec
                else:
                    continue
            else:
                flag = True
    lines.close()

    flag = False
    movie_dic = {}
    with open(movie_vec_dir, 'r', encoding='utf-8') as lines:
        for line in lines:
            if flag:
                movie_node = line.split(" ")[0]
                movie_node_vec = list(map(float, line.split(" ")[1:]))
                if movie_node not in movie_dic:
                    movie_dic[movie_node] = movie_node_vec
                else:
                    continue
            else:
                flag = True
    lines.close()
    with open(training_data_dir, 'w', encoding='utf-8') as writer:
        with open(ratings_dir, 'r', encoding='utf-8') as lines:
            for line in lines:
                userdID = line.split("::")[0]
                movieID = line.split("::")[1]
                rating = line.split("::")[2]
                if userdID in user_dic.keys() and movieID in movie_dic.keys():
                    user_dic[userdID].extend(movie_dic[movieID])
                    res = ""
                    for i in user_dic[userdID]:
                        res += str(i) + " "
                    writer.write(res + str(rating) + "\n")
        lines.close()
    writer.close()


def preprocess(user_vec_dir, movie_vec_dir, ratings_edge_dir, user_edge_dir, movie_edge_dir):
    flag = False
    user_dic = {}

    with open(user_vec_dir, 'r', encoding='utf-8') as lines:
        for line in lines:
            if flag:
                user_node = line.strip().split(" ")[0]
                user_node_vec = np.array(list(map(float, line.strip().split(" ")[1:])))
                if user_node not in user_dic:
                    user_dic[user_node] = user_node_vec
                else:
                    continue
            else:
                flag = True
    lines.close()

    flag = False
    movie_dic = {}
    with open(movie_vec_dir, 'r', encoding='utf-8') as lines:
        for line in lines:
            if flag:
                movie_node = line.strip().split(" ")[0]
                movie_node_vec = np.array(list(map(float, line.split(" ")[1:])))
                if movie_node not in movie_dic:
                    movie_dic[movie_node] = movie_node_vec
                else:
                    continue
            else:
                flag = True
    lines.close()

    user_user_dic = {}
    with open(user_edge_dir, 'r', encoding='utf-8') as lines:
        for line in lines:
            user1 = line.strip().split(" ")[0]
            user2 = line.strip().split(" ")[1]
            if user1 not in user_user_dic:
                user_user_dic[user1] = [user2]
            else:
                user_user_dic[user1].append(user2)
    lines.close()

    movie_movie_dic = {}
    with open(movie_edge_dir, 'r', encoding='utf-8') as lines:
        for line in lines:
            movie1 = line.strip().split(" ")[0]
            movie2 = line.strip().split(" ")[1]
            if movie1 not in movie_movie_dic:
                movie_movie_dic[movie1] = [movie2]
            else:
                movie_movie_dic[movie1].append(movie2)
    lines.close()

    train_x = np.zeros((128, 128))
    train_y = []
    user2movie_dic = {}
    movie2user_dic = {}
    with open(ratings_edge_dir, 'r', encoding='utf-8') as lines:
        for user_line in lines:
            userID_u = user_line.strip().split(" ")[0]
            movieID_u = user_line.strip().split(" ")[1]
            if userID_u not in user2movie_dic:
                user2movie_dic[userID_u] = [movieID_u]
            else:
                user2movie_dic[userID_u].append(movieID_u)
    lines.close()
    with open(ratings_edge_dir, 'r', encoding='utf-8') as lines:
        for movie_line in lines:
            userID_m = movie_line.strip().split(" ")[0]
            movieID_m = movie_line.strip().split(" ")[1]
            if movieID_m not in movie2user_dic:
                movie2user_dic[movieID_m] = [userID_m]
            else:
                movie2user_dic[movieID_m].append(userID_m)
    lines.close()
    with open(ratings_edge_dir, 'r', encoding='utf-8') as lines:
        for line in lines:
            userID = line.strip().split(" ")[0]
            movieID = line.strip().split(" ")[1]
            rating = line.strip().split(" ")[2]
            if userID in user_dic.keys() and movieID in movie_dic.keys():
                i = 0
                while i < 128:
                    if userID in user_user_dic.keys():
                        for con_user in user_user_dic[userID]:
                            if con_user in user2movie_dic.keys():  # 相邻的用户对该电影打过分
                                if movieID in user2movie_dic[con_user] and i<128:
                                    train_x[i] = list(map(float, user_dic[con_user]))
                                    i += 1
                    if movieID in movie_movie_dic.keys():
                        for con_movie in movie_movie_dic[movieID]:
                            if con_movie in movie2user_dic.keys():
                                if userID in movie2user_dic[con_movie] and i<128:   # 相邻的电影被该用户打过分
                                    train_x[i] = list(map(float, movie_dic[con_movie]))
                                    i += 1
                temp = [0, 0, 0, 0, 0]
                temp[int(rating) - 1] = 1
                train_y.append(temp)
    lines.close()

    x_train = np.array(train_x)
    y_train = np.array(train_y)
    return x_train, y_train


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    # print("x_len: ", data_len)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


if __name__ == "__main__":
    user_vec_dir = "./emb/users.emb"
    movie_vec_dir = "./emb/movies.emb"
    rating_edge_dir = "./graph/ratings.edge"
    rating_dir = "./data/ratings.txt"
    training_data_dir = "./data/traing.txt"
    user_dir = "./data/users.txt"
    user_edge_dir = "./graph/users.edge"
    movie_dir = "./data/movies.txt"
    movie_edge_dir = "./graph/movies.edge"
    # getUsersEdge(user_dir, user_edge_dir)
    # getMoviesEdge(movie_dir, movie_edge_dir)

    x_train, y_train = preprocess(user_vec_dir, movie_vec_dir, rating_edge_dir, user_edge_dir, movie_edge_dir)
    print("x_train: ", x_train)
    print("y_train: ", y_train)

    # getRatingEdge(user_vec_dir, movie_vec_dir, rating_dir, training_data_dir)
