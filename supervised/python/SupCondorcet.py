# UTF-8 
# python: 3.11.4

# 参考文献：The weighted Condorcet fusion in information retrieval（2013）
# Tancilon: 20231218

# 训练集数据输入格式：
## 文件1：train_rel_data: 1）csv文件格式 2）4列 Query | 0 | Item | Relevance
## 文件2：train_base_data： 1）csv文件格式 2）4列 Query | Voter name | Item Code | Item Rank
#      - Query 不要求是从1开始的连续整数
#      - Voter name 和 Item Code允许是字符串格式
# 定义算法的最终输出为csv文件格式：3列 Query | Item Code | Item Rank
#      - 注意输出的为排名信息，不是分数信息

# 测试集数据输入格式：
## 文件1：test_data: 1）csv文件格式 2）4列 Query | Voter name | Item Code | Item Rank
#      - Query 不要求是从1开始的连续整数
#      - Voter name 和 Item Code允许是字符串格式

# 数据输入可接受partial list，对于并列排名，采用随机排序的方式分出先后， Item Rank数值越小，排名越靠前
# 算法限制：
## 1.训练标签只能接受相关和不相关两种含义(0表示不相关，1表示相关）。
## 2.训练中的基础排序ranker与测试中的基础排序ranker一样

import numpy as np
import pandas as pd
import csv
import time
import matplotlib.pyplot as plt
import pickle

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

class LDA:
    
    """
    入口函数
    @param X: The type of X is np.array 
    @param y: The type of y is np.array
    """
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.target_classes = np.unique(y)
        # 每个类别的样本均值
        self.sub_mean = []
        for cls in self.target_classes:
            self.sub_mean.append(np.mean(X[list(y == cls)], axis=0))    
        # 所有类别的样本均值
        self.all_mean = np.mean(X, axis=0).reshape(1, X.shape[1])
        B = self.calculateB()
        S = self.calculateS()
        # 计算 S 的逆矩阵，乘以 B
        S_inv_B = np.linalg.inv(S).dot(B)
        # 计算对应的特征向量
        eig_vecs = self.sortEigenVectors(S_inv_B)
        return eig_vecs
    
    
    """
    计算类间散度矩阵 B
    """
    def calculateB(self):
        B = np.zeros((self.X.shape[1], self.X.shape[1]))
        for cls, mean in enumerate(self.sub_mean):
            # 每个类别的样本数
            n = self.X[self.y == cls].shape[0]
            mui_mu = mean.reshape(1, self.X.shape[1]) - self.all_mean
            B += n * np.dot(mui_mu.T, mui_mu)
        return B
        
    
    """
    计算类内散度矩阵 S
    """
    def calculateS(self):
        si_list = []
        for cls, mean in enumerate(self.sub_mean):
            si = np.zeros((self.X.shape[1], self.X.shape[1]))
            for xi in self.X[self.y == cls]:
                xi_mu = (xi - mean).reshape(1, self.X.shape[1])
                si += np.dot(xi_mu.T, xi_mu)
            si_list.append(si)
        S = np.zeros((self.X.shape[1], self.X.shape[1]))
        for si in si_list:
            S += si
        return S
    
    
    """
    计算最大特征向量
    """
    def sortEigenVectors(self, M):
        # 计算特征值，特征向量
        eig_values, eig_vectors = np.linalg.eig(M)
        # 排序找到最大特征值对应的特征向量
        idx = eig_values.argsort()[::-1]
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:,idx]
        return eig_vectors
    
    """
    初始化函数
    """
    def __init__(self):
        pass


class Supervised_Condorcet():

    def __init__(self):
        self.weights = None
        self.average_weight = None
        self.voter_name_reverse_mapping = None
        self.voter_name_mapping = None
        self.voter_num = None
        self.query_mapping = None
    """
    Return:
        r: 相关性标签, 1 * item一维Numpy数组, 数组内存放相关性
        R: item * voter二维Numpy数组, 数组内存放rank(排名), 如果voter_k 未给item_k排名, 则R[k,k] = 0
    """
    def convertToMatrix(self, base_data, rel_data = None):
        unique_items = base_data['Item Code'].unique()
        item_num = len(unique_items)
        item_mapping = {name: i for i, name in enumerate(unique_items)}
        R = np.zeros((item_num, self.voter_num))
        r = np.empty(item_num)

        for _, row in base_data.iterrows():
            voter_name = row['Voter Name']
            item_code = row['Item Code']
            item_rank = row['Item Rank']

            voter_index = self.voter_name_mapping[voter_name]
            item_index = item_mapping[item_code]
            R[item_index, voter_index] = item_rank

        if (rel_data is None):
            return R, item_mapping
        else:
            for _, row in rel_data.iterrows():
                item_code = row['Item Code']
                item_relevance = row['Relevance']

                item_index = item_mapping[item_code]
                r[item_index] = item_relevance
            return r, R, item_mapping


    def train(self, train_base_data, train_rel_data):
        train_base_data.columns = ['Query','Voter Name', 'Item Code', 'Item Rank']
        train_rel_data.columns= ['Query', '0', 'Item Code', 'Relevance']
        
        #print(train_rel_data)

        unique_queries = train_rel_data['Query'].unique()
        unique_voter_names = train_base_data['Voter Name'].unique()
        self.voter_num = len(unique_voter_names)
        # 建立整数到字符串的逆向映射
        self.voter_name_reverse_mapping = {i: name for i, name in enumerate(unique_voter_names)}
        self.voter_name_mapping = {v: k for k, v in self.voter_name_reverse_mapping.items()}


        ## 填充LDA相关类别数组X1
        X1 = []
        ## 填充LDA不相关类别数组X2
        X2 = []

        for query in tqdm(unique_queries):
            # 筛出当前query的数据
            base_data = train_base_data[train_base_data['Query'] == query]
            rel_data = train_rel_data[train_rel_data['Query'] == query]



            r, R, _ = self.convertToMatrix(base_data, rel_data)
            rel_items = np.where(r == 1)[0]
            irrel_items = np.where(r == 0)[0]


            for indexi in range(len(rel_items)):
                for indexj in range(len(irrel_items)):
                    itemi = rel_items[indexi]
                    itemj = irrel_items[indexj]
                    X1_features = [None] * self.voter_num
                    X2_features = [None] * self.voter_num
                    for k in range (self.voter_num):
                        if (R[itemi, k] == 0 and R[itemj, k] == 0):
                            X1_features[k] = 0
                            X2_features[k] = 0
                        elif (R[itemi, k] == 0 and R[itemj, k] != 0):
                            X1_features[k] = -1
                            X2_features[k] = 1
                        elif (R[itemi, k] != 0 and R[itemj, k] == 0):
                            X1_features[k] = 1
                            X2_features[k] = -1
                        else:
                            if (R[itemi, k] < R[itemj, k]):
                                X1_features[k] = 1
                                X2_features[k] = -1
                            else:
                                X1_features[k] = -1
                                X2_features[k] = 1

                    X1.append(X1_features)
                    X2.append(X2_features)

        X1 = np.array(X1)
        X2 = np.array(X2)
        y = np.zeros(X1.shape[0] + X2.shape[0])
        y[:X1.shape[0]] = 1
        y[X1.shape[0]:] = -1

        X = np.concatenate([X1, X2], axis=0)   
        

        # lda = LDA()
        # eig_vecs = lda.fit(X, y)
        # self.average_weight = eig_vecs[:, :1]

        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        self.average_weight = lda.coef_[0]

        # query_idx = self.query_mapping[query]
        # self.weights[query_idx, :] = eig_vecs[:, :1]
        # self.average_weight = np.mean(self.weights, axis = 0)


    def win(self, input_list, i, j):
        win_i = 0
        win_j = 0
        for k in range(self.voter_num):
            if (np.isnan(input_list[k, i]) and np.isnan(input_list[k, j])):
                continue
            elif (np.isnan(input_list[k, i]) and ~np.isnan(input_list[k, j])):
                win_j += self.average_weight[k]
            elif (~np.isnan(input_list[k, i]) and np.isnan(input_list[k, j])):
                win_i += self.average_weight[k]
            elif (input_list[k, i] < input_list[k, j]):
                win_i += self.average_weight[k]
            else:
                win_j += self.average_weight[k]
        return win_i, win_j


    def CondorcetAgg(self, input_list):
        num_items = input_list.shape[1]
        item_win_count = np.zeros(num_items)

        limit = np.sum(self.average_weight) / 2

        for i in range(num_items):
            for j in range(i + 1, num_items):
                # 项目对(i, j)中 i 赢了
                win_i, win_j = self.win(input_list, i, j)
                if (win_i > limit):
                    item_win_count[i] += 1
                # j 赢了
                if (win_j > limit):
                    item_win_count[j] += 1  
        
        first_row = item_win_count
        # 进行排序并返回排序后的列索引
        sorted_indices = np.argsort(first_row)[::-1]
        
        currrent_rank = 1
        result = np.zeros(num_items)
        for index in sorted_indices:
            result[index] = currrent_rank
            currrent_rank += 1

        return result

    
    def test(self, test_data, output):
        test_data.columns = ['Query','Voter Name', 'Item Code', 'Item Rank']
        # 获取唯一的Query值
        unique_queries = test_data['Query'].unique()

        with open(output, mode='w', newline='') as file:
            writer = csv.writer(file)

            for query in tqdm(unique_queries):
                # 筛选出当前Query的数据
                query_data = test_data[test_data['Query'] == query]

                # 创建空字典来保存Item Code和Voter Name的映射关系
                item_code_mapping = {}
                

                # 获取唯一的Item Code和Voter Name值，并创建索引到整数的映射
                unique_item_codes = query_data['Item Code'].unique()

                # 建立整数到字符串的逆向映射
                item_code_reverse_mapping = {i: code for i, code in enumerate(unique_item_codes)}

                # 生成字符串到整数的映射
                item_code_mapping = {v: k for k, v in item_code_reverse_mapping.items()}
                

                # 创建Voter Name*Item Code的二维Numpy数组，初始值为0
                num_items = len(unique_item_codes)
                input_list = np.full((self.voter_num, num_items), np.nan)

                #填充数组
                for index, row in query_data.iterrows():
                    voter_name = row['Voter Name']
                    item_code = row['Item Code']
                    item_rank = row['Item Rank']

                    voter_index = self.voter_name_mapping[voter_name]
                    item_index = item_code_mapping[item_code]

                    input_list[voter_index, item_index] = item_rank

                # 调用函数，获取排名信息
                rank = self.CondorcetAgg(input_list)

                # 将结果添加到result_df中
                for item_code_index, item_rank in enumerate(rank):   
                    item_code = item_code_reverse_mapping[item_code_index]
                    #result_df = result_df.append({'Query': query, 'Item Code': item_code, 'Rank': item_rank}, ignore_index=True)
                    new_row = [query, item_code, item_rank]
                    writer.writerow(new_row)
        
                

if __name__ == '__main__':
    print('Load training data...')
    start = time.perf_counter()

    # train_rel_loc = r'D:\RA_ReID\Topk_ReID\Market1501\top300_bdb-market1501-train-rel.csv'
    # train_base_loc = r'D:\RA_ReID\Topk_ReID\Market1501\top300_market1501_6workerlist_train_sim.csv'
    # test_loc = r'D:\RA_ReID\ReID_Dataset\Market1501\test\market1501_6workers.csv'
    # test_output_loc = r'result_supCond_market1501_6workers.csv'
    # save_model_loc = r'market_supCond.pkl'

    # train_rel_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold1\rel_train.csv'
    # train_base_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold1\rank_train.csv'
    test_loc = r"/mnt/disk2/dq/dataset/Top500_cuhk03detected_6workers.csv"
    test_output_loc = r'/mnt/disk2/dq/result/Top500_cuhk03detected_6workers_result_SupCondorcet.csv'
    # save_model_loc = r'test_supCond.pkl'

    # train_rel_data = pd.read_csv(train_rel_loc, header=None)

    # #print(train_rel_data)

    # train_base_data = pd.read_csv(train_base_loc, header=None)
    

    # model1 = Supervised_Condorcet()
    # model1.train(train_base_data, train_rel_data)

    # end = time.perf_counter()
    # print('train_time:', end - start)

    # print('Saving model...')

    # with open(save_model_loc, 'wb') as f:
    #     pickle.dump(model1, f)

    with open(r'SupCondorcet_cuhk03detected.pkl', 'rb') as f:
        model1 = pickle.load(f)


    test_data = pd.read_csv(test_loc, header=None) 
    print('Test...')
    model1.test(test_data, test_output_loc)


