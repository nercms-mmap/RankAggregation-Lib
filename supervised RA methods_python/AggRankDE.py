"""
UTF-8 
python: 3.11.4

参考文献: Aggregation of Rankings Using Metaheuristics in Recommendation Systems(2022)

Tancilon: 20231219


训练集数据输入格式：
文件1: train_rel_data: 
                        1)csv文件格式 
                        2)4列 Query | 0 | Item | Relevance
文件2: train_base_data: 
                        1) csv文件格式 
                        2)4列 Query | Voter name | Item Code | (Item Rank / Item Score)

- Query 不要求是从1开始的连续整数
- Voter name 和 Item Code允许是字符串格式
- train_base_data 默认情况下位Item Rank, 如果要输入Item Score, 设置模型参数type = score, 当设置为score时, 要求训练集和测试集都应该是score形式, 不能一个rank一个score

定义算法的最终输出为csv文件格式: 3列 Query | Item Code | Item Rank
    - 注意输出的为排名信息，不是分数信息

测试集数据输入格式：

文件1: test_data: 
                1) csv文件格式 
                2)4列 Query | Voter name | Item Code | (Item Rank / Item Score)
                - Query 不要求是从1开始的连续整数
                - Voter name 和 Item Code允许是字符串格式

其他细节：
        1) 数据输入最好为full lists, 对于partial list算法最后会给未排序的项目赋值0分
        2) Item Rank数值越小, 排名越靠前
        3) 训练集和测试集的Voter相同
"""

import numpy as np
import pandas as pd
import csv
import time
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pickle
from Evaluation import Evaluation

"""
参考文献: Using Differential Evolution in order to create a personalized list of recommended items(2020)
"""
def compute_AP(rank_list, rel_data, N, item_mapping):

    relevant_items = []
    for _, row in rel_data.iterrows():
        rel = row['Relevance']
        item_name = row['Item Code']
        if (rel > 0):
            item_id = item_mapping[item_name]
            relevant_items.append(item_id)

    """
    当不额外设置AP@N中N的数值时, 将N的数值取为所有相关的Item个数
    """
    relevant_num = len(relevant_items)
    if (N == None):
        N = relevant_num
    sumP = 0
    """
    或许有更加高效的写法
    """
    for n in range(N):
        count = 0
        if (rank_list[n] not in relevant_items):
            continue
        for i in range(N):
            if (rank_list[i] in relevant_items):
                count += 1
        p_n = count / (n + 1)
        sumP += p_n

    if (relevant_num == 0):
        return 0
    else:
        return sumP / relevant_num
            

"""
A: 初始排序构成的矩阵： item * voter 存储score
N: 在计算AP@N时N的取值, AP--->average precision
"""
def matrixFitnessFunction(A, P1, P1_fitness, P2, rel_list, N, item_mapping):

    evaluation = Evaluation()
    C2 = np.dot(A, P2)
    NP = P2.shape[1]
    """
    考虑每一个交叉变异后的新个体的性能
    """
    if (N is None):
        N = np.sum(rel_list > 0)

    for i in range(NP):
        item_column = C2[:, i]
        # 对item分数从高到低排序
        rank_list = np.argsort(item_column)[::-1] + 1

        ap = evaluation.compute_AP_r(rank_list, rel_list, N)
        # ap = compute_AP(rank_list, rel_data, N, item_mapping)
        """
        新个体更好,则原来的个体被淘汰
        """
        if (ap > P1_fitness[i]):
            P1_fitness[i] = ap
            P1[:, i] = P2[:, i] 

    return P1_fitness, P1


def DE(A, rel_data, max_iteration, CR, F, NP, D, N, item_mapping):


    rel_list = np.zeros(A.shape[0])
    for _, row in rel_data.iterrows():
        item_code = row['Item Code']
        item_rel = row['Relevance']
        item_id = item_mapping[item_code]
        rel_list[item_id] = item_rel


    """
    Generate Initial population matrix: voter * NP
    """
    voter_num = A.shape[1]
    P1 = np.empty((voter_num, NP))
    #随机数上界与下界的设置
    upperbound = 1
    lowerbound = 0
    for i in range(voter_num):
        for j in range(NP):
            P1[i, j] = random.uniform(lowerbound, upperbound)
    #计算初始种群的适应度
    C0 = np.dot(A, P1)
    fitness = np.empty(NP)
    evaluation = Evaluation()
    if (N is None):
        N = np.sum(rel_list > 0)

    for i in range(NP):
        item_column = C0[:, i]
        rank_list = np.argsort(item_column)[::-1] + 1
        fitness[i] = evaluation.compute_AP_r(rank_list, rel_list, N)
    """
    Evolution
    """
    P2 = np.empty((voter_num, NP))
    iteration = 0
    while (iteration < max_iteration):
        for i in range(NP):    
            """
            随机选择3个互不相同的个体
            """
            a = random.randint(0, NP - 1)
            while (a == i):
                a = random.randint(0, NP -1)

            b = random.randint(0, NP - 1)
            while (b == i or b == a):
                b = random.randint(0, NP - 1)
            
            c = random.randint(0, NP - 1)
            while (c == i or c == a or c == b):
                c = random.randint(0, NP - 1)

            """
            产生一个随机参数, 确保新个体中至少有一个变异基因
            """
            rnbr_i = random.randint(0, D - 1)

            for j in range(D):
                if (random.random() <= CR or j == rnbr_i):
                    """
                    变异
                    """
                    P2[j, i] = P1[j, a] + F * (P1[j, b] - P1[j, c])
                else:
                    """
                    交叉
                    """
                    P2[j, i] = P1[j, i]

        """
        评估与选择
        """
        fitness, P1 = matrixFitnessFunction(A, P1, fitness, P2, rel_list, N, item_mapping)
        iteration += 1

    """
    迭代结束后, 找到种群中最好的个体
    """
    max_index = np.argmax(fitness)
    return P1[:, max_index]


class AggRankDE():
    def __init__(self, NP = 50, max_iteration = 100, CR = 0.9, F = 0.5, type = 'rank', N = None):
        """
        hyper-parameters:
            NP: Population
            max_iteration: Number of Iterations
            CR: Crossover's Probability range [0, 1]
            F: Amplification Factor F , range [0, 2]
            type: rank or score --> default = rank (要么训练集和测试集都是rank, 要么训练集和测试集都为score)
            N: fitness function --> AP@N --> default = None 此时认为N是相关项目的数量
        """
        self.NP = NP
        self.max_iteration = max_iteration
        self.CR = CR
        self.F = F
        self.type = type
        self.N = N
        self.weights = None
        self.average_weight = None
        self.voter_name_reverse_mapping = None
        self.voter_name_mapping = None
        self.voter_num = None
        self.query_mapping = None

    def convertToMatrix(self, base_data):
        
        """
        转化后的item * voter 二维矩阵统一存储Item score
        """
        unique_items = base_data['Item Code'].unique()
        item_num = len(unique_items)
        item_mapping = {name: i for i, name in enumerate(unique_items)}
        A = np.zeros((item_num, self.voter_num))


        for _, row in base_data.iterrows():
            voter_name = row['Voter Name']
            item_code = row['Item Code']
            item_attribute = row['Item Attribute']

            voter_index = self.voter_name_mapping[voter_name]
            item_index = item_mapping[item_code]
            if (self.type == 'rank'):
                A[item_index, voter_index] = 1 / item_attribute
            elif (self.type == 'score'):
                A[item_index, voter_index] = item_attribute

        
        """
        对矩阵A按列进行归一化处理, 这里采用L2范数归一化
        (可能存在的问题: 不同的归一化技术是否会给之后的算法结果带来较大影响？)
        """
        column_norms = np.linalg.norm(A, axis=0)
        # normalized_A = A / column_norms
        # 防止除以零：给范数添加一个非常小的数
        column_norms_safe = column_norms + 1e-10
        normalized_A = A / column_norms_safe
        return normalized_A, item_mapping

        

    def train(self, train_base_data, train_rel_data):

        """
        Data process
        """
        train_base_data.columns = ['Query','Voter Name', 'Item Code', 'Item Attribute']
        train_rel_data.columns= ['Query', '0', 'Item Code', 'Relevance']

        unique_queries = train_rel_data['Query'].unique()
        unique_voter_names = train_base_data['Voter Name'].unique()
        self.voter_num = len(unique_voter_names)
        # 建立整数到字符串的逆向映射
        self.voter_name_reverse_mapping = {i: name for i, name in enumerate(unique_voter_names)}
        self.voter_name_mapping = {v: k for k, v in self.voter_name_reverse_mapping.items()}
        self.query_mapping = {name: i for i, name in enumerate(unique_queries)}

        self.weights = np.empty((len(unique_queries), self.voter_num))

        """
        Consider each query
        """
        #debug
        # debug = 1

        for query in tqdm(unique_queries):
            # print('debug:{0}'.format(debug))
            # debug += 1

            """
            筛出当前query的数据
            """
            base_data = train_base_data[train_base_data['Query'] == query]
            rel_data = train_rel_data[train_rel_data['Query'] == query]

            A, item_mapping = self.convertToMatrix(base_data)
            w = DE(A, rel_data, self.max_iteration, self.CR, self.F, self.NP, self.voter_num, self.N, item_mapping)
            

            """
            存储当前Query训练出的权重向量
            """
            query_index = self.query_mapping[query]
            self.weights[query_index, :] = w

        """
        下面算出针对所有Query的平均权重
        """
        self.average_weight = np.mean(self.weights, axis = 0)

    """
    using_average:
        1.选择使用的权重参数是否是平均权重
        2.当using_average = false 时, 用于测试集的query和训练集的query相同的情况, 主要针对推荐系统中排序聚合的情况
    """
    def test(self, test_data, test_output_loc, using_average_w = True):
        test_data.columns = ['Query','Voter Name', 'Item Code', 'Item Attribute']
        unique_test_queries = test_data['Query'].unique()

        with open(test_output_loc, mode='w', newline='') as file:
            writer = csv.writer(file)

            for query in tqdm(unique_test_queries):
                query_data = test_data[test_data['Query'] == query]          

                A, item_code_mapping = self.convertToMatrix(query_data)
                item_code_reverse_mapping = {v: k for k, v in item_code_mapping.items()}

                if (using_average_w == True):
                    score_list = np.dot(self.average_weight, A.T)
                else:
                    if (query not in self.query_mapping):
                        score_list = np.dot(self.average_weight, A.T)
                    else:
                        query_id = self.query_mapping[query]
                        score_list = np.dot(self.weights[query_id, :], A.T)
                
                rank_list = np.argsort(score_list)[::-1]
                
                for rank_index, item_id in enumerate(rank_list):
                    item_code = item_code_reverse_mapping[item_id]
                    new_row = [query, item_code, (rank_index + 1)]
                    writer.writerow(new_row)
                                
                

if __name__ == '__main__':
    print('Load training data...')
    start = time.perf_counter()

    """
    训练集和测试集的文件路径
    """
    # train_rel_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold1\rel_train.csv'
    # train_base_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold1\rank_train.csv'
    # test_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold1\rank_test.csv'
    # test_output_loc = r'C:\Users\2021\Desktop\Validate_SRA\Validate_AggRankDE\result_MQ2008-agg\result_AggRankDE_MQ2008-agg-Fold1.csv'

    """
    DukeMTMC_VideoReID
    """

    # train_rel_loc = r'D:\RA_ReID\ReID_Dataset\DukeMTMC_VideoReID\train\top300-PSTA-DukeMTMC_VideoReID-train-rel.csv'
    # train_base_loc = r'D:\RA_ReID\ReID_Dataset\DukeMTMC_VideoReID\train\top300-DukeMTMC_VideoReID_train_6worker_sim.csv'
    # test_loc = r'D:\RA_ReID\ReID_Dataset\DukeMTMC_VideoReID\test\DukeMTMC_VideoReID_6worker_sim.csv'
    # test_output_loc = r'result_AggRankDE_DukeMTMC_VideoReID_6worker_sim.csv'
    # save_model_loc = r'DukeMTMC_VideoReID_AggRankDE.pkl'


    """
    MARS
    """
    # train_rel_loc = r'D:\RA_ReID\ReID_Dataset\MARS\train\top300-PSTA-MARS-train-rel.csv'
    # train_base_loc = r'D:\RA_ReID\ReID_Dataset\MARS\train\top300_MARS_train_6worker_sim.csv'
    # test_loc = r'D:\RA_ReID\ReID_Dataset\MARS\test\MARS_6worker_sim.csv'
    # test_output_loc = r'result_AggRankDE_MARS_6worker_sim.csv'
    # save_model_loc = r'MARS_AggRankDE.pkl'

    train_rel_loc = r'/project/zhanghong/Tancilon/ReID/Market1501/top300_bdb-market1501-train-rel.csv'
    train_base_loc = r'/project/zhanghong/Tancilon/ReID/Market1501/top300_market1501_6workerlist_train_sim.csv'
    test_loc = r'/project/zhanghong/Tancilon/ReID/Market1501/market1501_6workers.csv'
    test_output_loc = r'/project/zhanghong/Tancilon/Result/ReID/result_AggRankDE_market1501_6workers.csv'
    save_model_loc = r'/project/zhanghong/Tancilon/Result/ReID/market1501_AggRankDE.pkl'

    """
    读文件
    """
    train_rel_data = pd.read_csv(train_rel_loc, header=None)
    train_base_data = pd.read_csv(train_base_loc, header=None)


    """
    Run
    """
    aggRankDE = AggRankDE(max_iteration=500, N = 10)
    aggRankDE.train(train_base_data, train_rel_data)

    end = time.perf_counter()
    print('train_time:', end - start)

    print('Saving model...')

    with open(save_model_loc, 'wb') as f:
        pickle.dump(aggRankDE, f)

    test_data = pd.read_csv(test_loc, header=None) 


    print('Test...')
    start = time.perf_counter()
    aggRankDE.test(test_data, test_output_loc,using_average_w=False)
    end = time.perf_counter()

    print('test_time:', end - start)




