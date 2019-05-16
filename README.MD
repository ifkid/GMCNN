# 图嵌入卷积神经网络

## 1、简介
基于导师关于多源异构数据的关联性分析的国自基项目衍生的电影打分预测的项目
## 2、相关数据集
[MovieLens-1m][1]
## 3、数据集介绍
用户节点: userID::gender::age::job::zip_code

电影节点: movieID::movie_title::movie_categories
## 4、文件目录
```shell
.
├── data # 数据
│   ├── movies.txt 
│   ├── users.txt 
│   ├── ratings.txt 
├── emb # 嵌入向量
│   ├── movies.emb 
│   ├── users.emd
├── graph # 不带权无向图
│   ├──users.edge 
│   ├──movies.edge 
│   ├──ratings.edge 
├── saver # 保存模型
├── tensorboard # 保存训练结果供可视化
│   ├──generateVec.py
│   ├──model.py
│   ├──node2vec.py
│   ├──run_model.py
│   ├──utils.py
├── requirements.txt
└── README.md
```
## 5、模型介绍
* 若用户与用户之间的性别、年龄层次、职业均相同，则认定它们之间强相关，它们之间存在无向边
* 若电影节点之间的电影类型相同，则认定它们之间存在强相关，它们之间存在无向边
* 上述结束得到两张不带权无向同构图，利用[Node2Vec算法][2]分别得到用户和电影节点的向量表示
* 将两向量相乘得到一个矩阵，输入到CNN网络中得到最终的分类结果(将回归问题看成一个五分类问题)


[1]: https://grouplens.org/datasets/movielens/1m/
[2]: https://github.com/eliorc/node2vec
