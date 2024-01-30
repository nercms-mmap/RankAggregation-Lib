"""
UTF-8 
python: 3.11.4
Tensorflow: 2.15.0

参考文献: CRF framework for supervised preference aggregation(2013)
Tancilon: 20231224


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
        1) 数据输入接受partial list
        2) Item Rank数值越小, 排名越靠前
"""

import numpy as np
import pandas as pd
import csv
import time
import tensorflow as tf
from itertools import permutations
from Evaluation import Evaluation
import math
from tqdm import tqdm
import pickle

class CRF():

    def __init__(self):
        self.weights = None
        self.average_weight = None
        self.voter_name_mapping = None
        self.voter_name_reverse_mapping = None
        self.voter_num = None

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
        r = np.zeros(item_num)

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

    def subsampleItems(self, base_data, rel_data, epsilon):
        """
        对rel_data进行Item抽样
        """
        # 确保每个不同的 Relevance 值至少有一个样本
        unique_relevance_samples = rel_data.groupby('Relevance').apply(lambda x: x.sample(1))

        #debug
        # if len(unique_relevance_samples) == 1:
        #     print("Warning...")
        #     print(rel_data)

        # 如果总样本数超过 epsilon，则进行抽样
        if len(unique_relevance_samples) > epsilon:
            sampled_data = unique_relevance_samples.sample(epsilon)
        else:
            sampled_data = unique_relevance_samples

        # 确保 Item Code 互不相同并且样本数为N
        while (len(sampled_data['Item Code'].unique()) < epsilon):
            additional_samples = rel_data.sample(1)
            sampled_data = pd.concat([sampled_data, additional_samples]).drop_duplicates(subset='Item Code').reset_index(drop=True)

        """
        根据抽样结果处理base_data
        """

        # 获取 sampled_data 中的 'Item Code' 列的唯一值
        sampled_item_codes = sampled_data['Item Code'].unique()
        # 过滤掉 base_data 中 'Item Code' 列中不在 sampled_item_codes 中的行
        filtered_base_data = base_data[base_data['Item Code'].isin(sampled_item_codes)]

        return filtered_base_data, sampled_data
    
    """
    y: 1 * item一维Numpy数组, 数组内存放项目排名
    r: 相关性标签, 1 * item一维Numpy数组, 数组内存放相关性
    """
    def compute_loss(self, y, r, loss_cut_off):
        # 如果没有手动设置loss_cut_off, 修改为所有相关性文档的数量
        if (loss_cut_off is None):
             loss_cut_off = np.sum(r > 0)
        evaluation = Evaluation()
        ndcg = evaluation.compute_ndcg_r(y, r, loss_cut_off)
        return 1 - ndcg
        

    # 注意theta是tf中的对象
    def commpute_negative_engergy(self, y, R, theta):
        item_num = len(y)
        voter_num = R.shape[1]
        # i: 考虑所有的排名
        negative_energy = 0.0
        for i in range(1, item_num + 1):
            item_info = 0.0
            for k in range(voter_num):
                # item_id = np.where(y == i)[0]
                item_id = np.argmax(y == i)
                if (R[item_id, k] == 0):
                    item_info += theta[3 * k]
                else:
                    for j in range(item_num):
                        if (j == item_id or R[item_id, k] == 0 or R[j, k] == 0):
                            continue
                        if (R[item_id, k] < R[j, k]):
                            item_info += theta[3 * k + 1] * (R[j, k] - R[item_id, k]) / np.max(R[:, k])
                        if (R[j, k] < R[item_id, k]):
                            item_info -= theta[3 * k + 2] * (R[item_id, k] - R[j, k]) / np.max(R[:, k])
            item_info = item_info / tf.math.log(tf.cast(i + 1, tf.float64))
            negative_energy += item_info
        
        negative_energy = negative_energy / (item_num * item_num)   
        return negative_energy




    """
    Param:
        alpha: learning rate
        epsilon: cut-off ,该参数的取值至少要大于数据集中不同相关性标签的种类数(eg:如果相关性标签只有0和1两种取值,则epsilon至少大于2)
        loss_type_cut: ndcg@k 中k的取值, 若为None, 则赋值为所有相关性文档的数量
    """
    def train(self, train_base_data, train_rel_data, alpha = 0.01, epsilon = 5, epoch = 300, loss_cut_off = None):
        """
        Data process
        """
        train_base_data.columns = ['Query','Voter Name', 'Item Code', 'Item Rank']
        train_rel_data.columns= ['Query', '0', 'Item Code', 'Relevance']
        unique_queries = train_rel_data['Query'].unique()
        unique_voter_names = train_base_data['Voter Name'].unique()
        self.voter_num = len(unique_voter_names)
        self.voter_name_reverse_mapping = {i: name for i, name in enumerate(unique_voter_names)}
        self.voter_name_mapping = {v: k for k, v in self.voter_name_reverse_mapping.items()}

        """
        Initialize weights
        """
        self.weights = np.zeros(3 * self.voter_num)

        """
        Repeat CRF optimization
        """

        for epo in range(epoch):
            #debug
            # print("epo={0}".format(epo))
            for query in tqdm(unique_queries):
                """
                筛出当前query的数据
                """

                base_data = train_base_data[train_base_data['Query'] == query]
                rel_data = train_rel_data[train_rel_data['Query'] == query]
                unique_items = base_data['Item Code'].unique()
                if (len(unique_items) > epsilon):
                    if (len(rel_data) < epsilon):
                        continue
                    subs_base_data, subs_rel_data = self.subsampleItems(base_data, rel_data, epsilon)
                    r, R, _ = self.convertToMatrix(subs_base_data, subs_rel_data)
                else:
                    r, R, _ = self.convertToMatrix(base_data, rel_data)

                """
                计算梯度
                """ 
                theta = tf.Variable(self.weights)

                #debug
                # print("query is {0}".format(query))


                # 使用 tf.GradientTape() 来记录计算过程

                with tf.GradientTape() as tape:

                    # 初始化 y
                    objective = 0.0
                    # 枚举每一种可能的排序
                    initial_perm = np.empty(len(r))
                    for i in range(len(r)):
                        initial_perm[i] = i + 1;
                    all_permutations = permutations(initial_perm)
                    # y: 1 * item一维Numpy数组，数组内存放项目排名
                    sum_exp_negative_energy = 0.0
                    for perm in all_permutations:
                        y = np.array(perm)
                        loss = self.compute_loss(y, r, loss_cut_off)
                        negative_energy = self.commpute_negative_engergy(y, R, theta)
                        objective += loss * tf.exp(negative_energy)
                        sum_exp_negative_energy += tf.exp(negative_energy)
                    objective = objective / sum_exp_negative_energy

                # 计算函数 y 相对于向量 x 的梯度
                grad = tape.gradient(objective, theta)

                self.weights = self.weights - alpha * grad.numpy()

    def test(self, test_data, output):
        test_data.columns = ['Query','Voter Name', 'Item Code', 'Item Rank'] 
        unique_test_queries = test_data['Query'].unique()


        with open(output, mode='w', newline='') as file:
            writer = csv.writer(file)
            for query in tqdm(unique_test_queries):
                query_data = test_data[test_data['Query'] == query]
                R, item_code_mapping = self.convertToMatrix(query_data)
                item_code_reverse_mapping = {v: k for k, v in item_code_mapping.items()}
                item_num = R.shape[0]
                score_list = np.empty(item_num)

                for i in range(item_num):
                    score_i = 0.0
                    for k in range(self.voter_num):
                        if (R[i, k] == 0):
                            score_i -= self.weights[3 * k]
                        else:
                            max_rank = np.max(R[:, k])
                            score_i -= self.weights[3 * k + 1] * ((max_rank - R[i, k]) * (max_rank + 1 - R[i, k]) / (max_rank * 2))
                            score_i += self.weights[3 * k + 2] * ((R[i, k] - 1) * R[i, k] / (2 * max_rank))
                    score_list[i] = score_i

                rank_list = np.argsort(score_list)
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
    # train_rel_loc = r'/mnt/disk6new/dq/Market1501/bdb-market1501-train-rel.csv'
    # train_base_loc = r'/mnt/disk6new/dq/Market1501/market1501_6workerlist_train_sim.csv'
    # test_loc = r'/mnt/disk6new/dq/Market1501/market1501_6workers.csv'
    # test_output_loc = r'/mnt/disk6new/dq/result/Market1501/result_CRF_market1501_6workers.csv'
    # save_model_loc = r'/mnt/disk6new/dq/result/Market1501/Market_CRF.pkl'


    # train_rel_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold1\rel_train.csv'
    # train_base_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold1\rank_train.csv'
    # test_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold1\rank_test.csv'
    # test_output_loc = r'crf_out_test.csv'
    # save_model_loc = r'test_CRF.pkl'

    """
    DukeMTMC_VideoReID
    """

    # train_rel_loc = r'D:\RA_ReID\ReID_Dataset\DukeMTMC_VideoReID\train\top300-PSTA-DukeMTMC_VideoReID-train-rel.csv'
    # train_base_loc = r'D:\RA_ReID\ReID_Dataset\DukeMTMC_VideoReID\train\top300-DukeMTMC_VideoReID_train_6worker_sim.csv'
    # test_loc = r'D:\RA_ReID\ReID_Dataset\DukeMTMC_VideoReID\test\DukeMTMC_VideoReID_6worker_sim.csv'
    # test_output_loc = r'result_CRF_DukeMTMC_VideoReID_6worker_sim.csv'
    # save_model_loc = r'DukeMTMC_VideoReID_CRF.pkl'



    # train_rel_loc = r'/project/zhanghong/Tancilon/MovieLens1m/m1m-top20-train-rel.csv'
    # train_base_loc = r'/project/zhanghong/Tancilon/MovieLens1m/m1m-top20-train.csv'
    # test_loc = r'/project/zhanghong/Tancilon/MovieLens1m/mlm-top20-test.csv'
    # test_output_loc = r'/project/zhanghong/Tancilon/Result/m1m/result-CRF-mlm-top20-test.csv'
    # save_model_loc = r'/project/zhanghong/Tancilon/Result/m1m/m1m_CRF.pkl'

    # """
    # 读文件
    # """
    # train_rel_data = pd.read_csv(train_rel_loc, header=None)
    # train_base_data = pd.read_csv(train_base_loc, header=None)
    # test_data = pd.read_csv(test_loc, header=None) 

    # """
    # Run
    # """
    # crf = CRF()
    # crf.train(train_base_data, train_rel_data, epoch=1, epsilon=4)

    # end = time.perf_counter()
    # print('train_time:', end - start)

    # print('Saving model...')
    # with open(save_model_loc, 'wb') as f:
    #     pickle.dump(crf, f)

    # print('Test...')
    # start = time.perf_counter()
    # crf.test(test_data, test_output_loc)
    # end = time.perf_counter()
    # print('test_time:', end - start)



    with open(r'D:\RA_ReID\Evaluation\m1m_result\m1m_CRF.pkl', 'rb') as f:
        model = pickle.load(f)
    print(model.weights)