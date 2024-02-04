"""
semi-supervised

UTF-8 
python: 3.11.4

参考文献: Semi-supervised Ranking Aggregation(2008)
Tancilon: 20240118


训练集数据输入格式：
文件1: train_rel_data: 
                        1)csv文件格式 
                        2)4列 Query | 0 | Item | Relevance
文件2: train_base_data: 
                        1) csv文件格式 
                        2)4列 Query | Voter name | Item Code | Item Rank

- Query 不要求是从1开始的连续整数
- Voter name 和 Item Code允许是字符串格式


定义算法的最终输出为csv文件格式: 3列 Query | Item Code | Item Rank
    - 注意输出的为排名信息，不是分数信息

测试集数据输入格式：

文件1: test_data: 
                1) csv文件格式 
                2)4列 Query | Voter name | Item Code | Item Rank
                - Query 不要求是从1开始的连续整数
                - Voter name 和 Item Code允许是字符串格式

其他细节：
        1) 数据输入接受Full list, 对Partial list的处理采用排在最后一名的方式
        2) Item Rank数值越小, 排名越靠前
"""


import cvxpy as cp
import numpy as np
import pandas as pd
import pickle
import time
import csv

from tqdm import tqdm
from scipy.stats import kendalltau



class SSRA():
    def __init__(self):
        self.weights = None
        self.average_weight = None
        self.voter_name_mapping = None
        self.voter_name_reverse_mapping = None
        self.voter_num = None
        self.is_partial_list = None
        self.query_mapping = None
        """
        rank_base_data_matrix: voter * item
        """
        self.rank_base_data_matrix = None

    def partialToFull(self, rank_base_data_matrix):
        # 扩充为full list的方式是将未排序的项目全部并列放在最后一名
        num_voters = rank_base_data_matrix.shape[0]

        for k in range(num_voters):
            if np.isnan(rank_base_data_matrix[k]).all():
                # 处理全为 NaN 的切片
                rank_base_data_matrix[k] = np.nan_to_num(rank_base_data_matrix[k], nan = rank_base_data_matrix.shape[1])
            else:
                max_rank = np.nanmax(rank_base_data_matrix[k])
                rank_base_data_matrix[k] = np.nan_to_num(rank_base_data_matrix[k], nan = max_rank + 1)

        return rank_base_data_matrix

    """
    return:
        score_base_data_matrix: item * voter 存储Borda分数
        rel_data_matrix: 1 * item 存储item的相关性
    """
    def convertToMatrix(self, base_data, rel_data = None):
        unique_items = base_data['Item Code'].unique()
        item_num = len(unique_items)
        item_mapping = {name: i for i, name in enumerate(unique_items)}
        rank_base_data_matrix = np.full((self.voter_num, item_num), np.nan)
        score_base_data_matrix = np.empty((self.voter_num, item_num))
        rel_data_matrix = np.empty(item_num)

        for _, row in base_data.iterrows():
            voter_name = row['Voter Name']
            item_code = row['Item Code']
            item_rank = row['Item Rank']

            voter_index = self.voter_name_mapping[voter_name]
            item_index = item_mapping[item_code]
            rank_base_data_matrix[voter_index, item_index] = item_rank

        if (self.is_partial_list == True):
            rank_base_data_matrix = self.partialToFull(rank_base_data_matrix)
        
        self.rank_base_data_matrix = rank_base_data_matrix

        for k in range(self.voter_num):
            max_rank = np.max(rank_base_data_matrix[k, :])
            for i in range(item_num):
                score_base_data_matrix[k, i] = max_rank - rank_base_data_matrix[k, i] + 1

        if (rel_data is None):
            score_base_data_matrix = score_base_data_matrix.T
            return score_base_data_matrix, item_mapping
        else:
            for _, row in rel_data.iterrows():
                item_code = row['Item Code']
                item_relevance = row['Relevance']

                item_index = item_mapping[item_code]
                rel_data_matrix[item_index] = item_relevance

            score_base_data_matrix =  score_base_data_matrix.T
            return score_base_data_matrix, rel_data_matrix, item_mapping



    def get_norm_similarity(self):
        p = np.zeros(self.voter_num)
        for i in range(self.voter_num):
            for j in range(i + 1, self.voter_num):
                tau, _ = kendalltau(self.rank_base_data_matrix[i, :], self.rank_base_data_matrix[j, :])
                if np.isnan(tau):
                    normalized_tau = 0
                else:
                    normalized_tau = (1 - tau) / 2
                p[i] += (1 / self.voter_num) * normalized_tau
                p[j] += (1 / self.voter_num) * normalized_tau

        return p

    def borda_count(self, rankings):
        num_voters, num_items = rankings.shape
        scores = np.zeros(num_items)
        for i in range(num_items):
            scores[i] = np.sum(num_items - rankings[:, i])
        return scores

    def borda_weighted_graph(self, scores):
        num_items = len(scores)
        adjacency_matrix = np.zeros((num_items, num_items))

        max_differ = np.max(scores) - np.min(scores)
        
        for i in range(num_items):
            for j in range(i + 1, num_items):
                if i != j:
                    # 使用分数差的绝对值的倒数作为权重
                    adjacency_matrix[i, j] = 1.0 / (np.abs(scores[i] - scores[j]) + 1e-5)

                    # adjacency_matrix[i, j] = max_differ - np.abs(scores[i] - scores[j])
                    adjacency_matrix[j, i] = adjacency_matrix[i, j]

        return adjacency_matrix


    def get_Laplacian(self):
        # 计算Borda count分数
        borda_scores = self.borda_count(self.rank_base_data_matrix)

        # 创建基于Borda count的邻接矩阵
        W = self.borda_weighted_graph(borda_scores)
        row_sums = np.sum(W, axis=1)
        D = np.diag(row_sums)
        L = D - W


        return L


    def regularize_to_positive_semidefinite_matrix(self, arr, regularization_param):
        n = arr.shape[0]  # 获取矩阵的维度
        I = np.identity(n)  # 创建单位矩阵

        # 添加正则化参数到对角元素
        arr_reg = arr + regularization_param * I

        # 计算修正的Cholesky分解
        L = np.linalg.cholesky(arr_reg)

        # 构建半正定矩阵
        positive_semidefinite_matrix = np.dot(L, L.T)
        
        return positive_semidefinite_matrix


    """
    alpha, beta: Hyperparameters, typically ranging from 0.01 to 0.1
    constraints_rate: The proportion of supervisory information used, ranging from 0 to 1. If it is 1, the method becomes fully supervised
    """
    def train(self, train_base_data, train_rel_data, alpha = 0.03, beta = 0.1, constraints_rate = 0.3, is_partial_list = True):
        """
        Data process
        """
        train_base_data.columns = ['Query','Voter Name', 'Item Code', 'Item Rank']
        train_rel_data.columns= ['Query', '0', 'Item Code', 'Relevance']
        unique_queries = train_rel_data['Query'].unique()
        unique_voter_names = train_base_data['Voter Name'].unique()
        self.is_partial_list = is_partial_list
        self.voter_num = len(unique_voter_names)
        self.voter_name_reverse_mapping = {i: name for i, name in enumerate(unique_voter_names)}
        self.voter_name_mapping = {v: k for k, v in self.voter_name_reverse_mapping.items()}
        self.query_mapping = {name: i for i, name in enumerate(unique_queries)}
        self.weights = np.zeros((len(unique_queries), self.voter_num))


        # count = 1

        for query in tqdm(unique_queries):

            # if (count == 9):
            #     print('debug....')
            # count += 1


            base_data = train_base_data[train_base_data['Query'] == query]
            rel_data = train_rel_data[train_rel_data['Query'] == query]

            R, rel_data_matrix, _ = self.convertToMatrix(base_data, rel_data)

            rel_items = np.where(rel_data_matrix >= 1)[0]
            irrel_items = np.where(rel_data_matrix <= 0)[0]
            if (len(rel_items) == 0):
                continue
            rel_nums = int(len(rel_items) * constraints_rate)
            if (rel_nums) < 1:
                rel_nums = 1

            p = self.get_norm_similarity()

            # print(p)

            L_prime = self.get_Laplacian()



            # 定义优化变量
            w = cp.Variable(p.shape[0])

            matrix = R.T @ L_prime @ R
            if not np.all(np.linalg.eigvals(matrix) >= 0):
                regularization_param = 0.01
                matrix = self.regularize_to_positive_semidefinite_matrix(matrix, regularization_param)
            
            
            # print(np.linalg.eigvals(matrix))
            # if not np.all(np.linalg.eigvals(matrix) >= 0):
            #     print("debug...")

            matrix = cp.psd_wrap(matrix)
            # 定义目标函数
            objective = cp.Minimize(cp.norm(w - p)**2 + alpha * cp.quad_form(w, matrix) + (beta/2) * cp.norm(w)**2)


            # 定义约束条件
            constraints = []
            for indexi in range(rel_nums):
                for indexj in range(len(irrel_items)):
                    i = rel_items[indexi]
                    j = irrel_items[indexj]
                    constraints.append(R[i, :] @ w.T - R[j, :] @ w.T >= 1)

            constraints.append(w >= 0)
            constraints.append(cp.sum(w) == 1)


            # 设置问题
            problem = cp.Problem(objective, constraints)

            # 求解问题
            problem.solve(solver = cp.SCS)

            query_idx = self.query_mapping[query]   

            self.weights[query_idx, :] = w.value

        # 创建一个布尔索引数组，标识出所有非全零的行
        non_zero_rows = np.any(self.weights != 0, axis=1)
        # 检查每行是否不含NaN
        not_nan_rows = ~np.any(np.isnan(self.weights), axis=1)

        # 使用此布尔索引数组来选择参与平均值计算的行
        filtered_weights = self.weights[non_zero_rows & not_nan_rows]

        # 计算这些选定行的平均值
        self.average_weight = np.mean(filtered_weights, axis=0)

        

    def test(self, test_data, test_output_loc, using_average_w = True):
        test_data.columns = ['Query','Voter Name', 'Item Code', 'Item Rank']    
        unique_test_queries = test_data['Query'].unique()
         # 创建一个空的DataFrame来存储结果

        with open(test_output_loc, mode='w', newline='') as file:
            writer = csv.writer(file)

            for query in tqdm(unique_test_queries):
                query_data = test_data[test_data['Query'] == query] 
                query_data_matrix, item_code_mapping = self.convertToMatrix(query_data)
                query_data_matrix = query_data_matrix.T
                item_code_reverse_mapping = {v: k for k, v in item_code_mapping.items()}

                if (using_average_w == True):
                    score_list = np.dot(self.average_weight, query_data_matrix)
                else:
                    if (query not in self.query_mapping):
                        score_list = np.dot(self.average_weight, query_data_matrix)
                    else:
                        query_id = self.query_mapping[query]
                        score_list = np.dot(self.weights[query_id, :], query_data_matrix)

                rank_list = np.argsort(score_list)[::-1]
                for rank_index, item_id in enumerate(rank_list):
                    item_code = item_code_reverse_mapping[item_id]
                    new_row = [query, item_code, (rank_index + 1)]
                    writer.writerow(new_row) 


if __name__ == '__main__':
    print('Load training data...')
    start = time.perf_counter()  

    # train_rel_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold1\rel_train.csv'
    # train_base_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold1\rank_train.csv'
    # test_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold1\rank_test.csv'
    # test_output_loc = r'MQ2008-agg-SSRA-test.csv'
    # save_model_loc = r'MQ2008-agg-SSRA.pkl'

    # train_rel_loc = r'D:\RA_ReID\Topk_ReID\CHUK03_detected\top300_bdb-cuhk03detected-train-rel.csv'
    # train_base_loc = r'D:\RA_ReID\Topk_ReID\CHUK03_detected\top300_cuhk03detected_6workerlist_train_sim.csv'
    # test_loc = r'D:\RA_ReID\Topk_ReID\CHUK03_detected\cuhk03detected_6workers.csv'
    # test_output_loc = r'D:\RA_ReID\Topk_result\detected\result_SSRA_cuhk03detected_6workers.csv'
    # save_model_loc = r'D:\RA_ReID\Topk_result\detected\SSRA_cuhk03detected.pkl'

    train_rel_loc = r'C:\Users\2021\Desktop\MovieLens\m1m-top20-train-rel.csv'
    train_base_loc = r'C:\Users\2021\Desktop\MovieLens\m1m-top20-train.csv'
    test_loc = r'C:\Users\2021\Desktop\MovieLens\mlm-top20-test.csv'
    test_output_loc = r'D:\RA_ReID\Evaluation\m1m_result\result-SSRA-m1m-top20-test-0.5.csv'
    save_model_loc = r'D:\RA_ReID\Evaluation\m1m_result\SSRA-m1m-top20-test-0.5.pkl'

    """
    读文件
    """
    train_rel_data = pd.read_csv(train_rel_loc, header=None)
    train_base_data = pd.read_csv(train_base_loc, header=None)


    """
    Run
    """
    ssra = SSRA()
    ssra.train(train_base_data, train_rel_data, constraints_rate=0.5)

    end = time.perf_counter()
    print('train_time:', end - start)

    print('Saving model...')
    with open(save_model_loc, 'wb') as f:
        pickle.dump(ssra, f)

    test_data = pd.read_csv(test_loc, header=None) 
    print('Test...')
    ssra.test(test_data, test_output_loc)


    # with open(r'C:\Users\2021\Desktop\Supervised RA\MQ2008-agg-SSRA.pkl', 'rb') as f:
    #     model = pickle.load(f)
    # print(model.average_weight)