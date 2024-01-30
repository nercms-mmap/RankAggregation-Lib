"""
存储各种评估指标的代码实现
Tancilon: 20231219
"""
import numpy as np
import pandas as pd
import math
import scipy.io

class Evaluation():
    def __init__(self) -> None:
        pass


    """
    Data Process
    """

    def covert_pd_to_csv(self, query_test_data, query_rel_data):
        unique_items = query_test_data['Item Code'].unique()

        item_num = len(unique_items)
        item_mapping = {name: i for i, name in enumerate(unique_items)}
        ra_list = np.zeros(item_num)
        rel_list = np.zeros(item_num)

        for _, row in query_test_data.iterrows():
            item_code = row['Item Code']
            item_rank = row['Item Rank']

            item_id = item_mapping[item_code]
            ra_list[item_id] = item_rank

        for _, row in query_rel_data.iterrows():
            item_code = row['Item Code']
            item_rel = row['Relevance']

            if item_code not in item_mapping:
                continue

            item_id = item_mapping[item_code]
            rel_list[item_id] = item_rel
        
        return ra_list, rel_list


    """
    Compute precision
    Precision is the proportion of the retrieved documents that are relevant.
        Precision = r / n
    where,
        r is the number of retrieved relevant documents;
        n is the number of retrieved documents.

    compute_P_s:
            score_list: 1 * item, 一维Numpy数组, 数组内存放分数
            rel_list: 1 * item, 一维Numpy数组, 数组内存放相关性分数
    
    compute_P_r:
            list: 1 * item, 一维Numpy数组, 数组内存放排名
            rel_list: 1 * item, 一维Numpy数组, 数组内存放相关性分数
    
    compute_AP_r:
            list: 1 * item, 一维Numpy数组, 数组内存放排名
            rel_list: 1 * item, 一维Numpy数组, 数组内存放相关性分数

    compute_AP_s:
            score_list: 1 * item, 一维Numpy数组, 数组内存放分数
            rel_list: 1 * item, 一维Numpy数组, 数组内存放相关性分数
            
    eval_map:
            用于对算法输出的结果进行评测
            test_data: 
                - csv文件格式
                - Query | Item Code | Item Rank
            rel_data:
                - csv文件格式
                - Query | 0 | Item | Relevance
    """

    def compute_P_s(self, score_list, rel_list, topk):
        if (topk <= 0 or topk > len(rel_list)):
            topk = len(rel_list)
        # 将分数转化为排名，排名从0开始
        rank_list = np.argsort(score_list)[::-1]
        r = 0

        for k in range(topk):
            item_idx = rank_list[k]
            if (rel_list[item_idx] > 0):
                r += 1
        
        return r / topk
    
    def compute_P_r(self, list, rel_list, topk):
        rank_list = np.argsort(list)
        r = 0
        if (topk > len(rel_list)):
            # print("Warning: Calculate precision metrics where topk is greater than the number of items")
            topk = len(rel_list)
        
        for k in range(topk):
            item_idx = rank_list[k]
            if (rel_list[item_idx] > 0):
                r += 1
        return r / topk

    def compute_R_r(self, list, rel_list, topk):
        rank_list = np.argsort(list)
        if (topk > len(rel_list)):
            # print("Warning: Calculate precision metrics where topk is greater than the number of items")
            topk = len(rel_list)

        for k in range(topk):
            item_idx = rank_list[k]
            if (rel_list[item_idx] > 0):
                return 1
        
        return 0
    
    def compute_R_s(self, score_list, rel_list, topk):
        if (topk <= 0 or topk > len(rel_list)):
            topk = len(rel_list)
        # 将分数转化为排名，排名从0开始
        rank_list = np.argsort(score_list)[::-1]
        for k in range(topk):
            item_idx = rank_list[k]
            if (rel_list[item_idx] > 0):
                return 1
            
        return 0



    def compute_AP_r(self, list, rel_list, topk):
        total_r = np.sum(rel_list > 0)
        ap = 0.0
        for k in range(1, topk + 1):
            item_id = np.argmax(list == k)
            if (rel_list[item_id] > 0):
                p_k = self.compute_P_r(list, rel_list, k)
                ap += p_k / total_r   
        return ap

    def compute_AP_s(self, score_list, rel_list, topk):
        sorted_indices = np.argsort(score_list)[::-1]
        para_list = np.argsort(sorted_indices) + 1
        ap = self.compute_AP_r(para_list, rel_list, topk)
        return ap






    def eval_map(self, test_data, rel_data, topk = None):
        test_data.columns = ['Query', 'Item Code', 'Item Rank']
        rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']

        unique_queries = test_data['Query'].unique()
        sum_ap = 0.0
        for query in unique_queries:
            query_test_data = test_data[test_data['Query'] == query]
            query_rel_data = rel_data[rel_data['Query'] == query]
            ra_list, rel_list = self.covert_pd_to_csv(query_test_data, query_rel_data)
            if (topk is None):
                topk = len(ra_list)
            sum_ap += self.compute_AP_r(ra_list, rel_list, topk)
        
        return sum_ap / len(unique_queries)
    




    def eval_map_matlab(self, test_path, rel_path, test_data_name, test_rel_name, type = None, topk = 10):
        test_mat = scipy.io.loadmat(test_path)
        rel_mat = scipy.io.loadmat(rel_path)
        # with h5py.File(rel_path, 'r') as file:
        #     test_rel = file['testRelevanceMatrix'][()]
        # rel_data = np.transpose(test_rel, axes=[1, 0]) 

        test_data = test_mat[test_data_name]
        rel_data = rel_mat[test_rel_name]


        # print(test_data.shape[0])
        # print(test_data.shape[1])
        # print(rel_data.shape[0])
        # print(rel_data.shape[1])

        if (type == 'rank'):
            sum_ap = 0.0
            for query in range(test_data.shape[0]):
                sum_ap += self.compute_AP_r(test_data[query,:], rel_data[query,:], topk)
            
            return sum_ap / test_data.shape[0]
        
        elif(type == 'score'):
            sum_ap = 0.0
            for query in range(test_data.shape[0]):
                sum_ap += self.compute_AP_s(test_data[query,:], rel_data[query,:], topk)
            
            return sum_ap / test_data.shape[0]




    def eval_r1_matlab(self, test_path, rel_path, test_data_name, test_rel_name, type = None):
        test_mat = scipy.io.loadmat(test_path)
        rel_mat = scipy.io.loadmat(rel_path)

        # with h5py.File(rel_path, 'r') as file:
        #     test_rel = file['testRelevanceMatrix'][()]
        # rel_data = np.transpose(test_rel, axes=[1, 0]) 


        test_data = test_mat[test_data_name]
        rel_data = rel_mat[test_rel_name]


        # print(test_data.shape[0])
        # print(test_data.shape[1])
        # print(rel_data.shape[0])
        # print(rel_data.shape[1])

        if (type == 'rank'):
            sum_r = 0.0
            for query in range(test_data.shape[0]):
                sum_r += self.compute_P_r(test_data[query,:], rel_data[query,:], 1)
            
            return sum_r / test_data.shape[0]
        
        elif(type == 'score'):
            sum_r = 0.0
            for query in range(test_data.shape[0]):
                sum_r += self.compute_P_s(test_data[query,:], rel_data[query,:], 1)
            
            return sum_r / test_data.shape[0]
        

    def eval_r_matlab(self, test_path, rel_path, test_data_name, test_rel_name, type = None, topk = 1):
        test_mat = scipy.io.loadmat(test_path)
        rel_mat = scipy.io.loadmat(rel_path)

        # with h5py.File(rel_path, 'r') as file:
        #     test_rel = file['testRelevanceMatrix'][()]
        # rel_data = np.transpose(test_rel, axes=[1, 0]) 


        test_data = test_mat[test_data_name]
        rel_data = rel_mat[test_rel_name]


        # print(test_data.shape[0])
        # print(test_data.shape[1])
        # print(rel_data.shape[0])
        # print(rel_data.shape[1])

        if (type == 'rank'):
            sum_r = 0.0
            for query in range(test_data.shape[0]):
                sum_r += self.compute_R_r(test_data[query,:], rel_data[query,:], topk)
            
            return sum_r / test_data.shape[0]
        
        elif(type == 'score'):
            sum_r = 0.0
            for query in range(test_data.shape[0]):
                sum_r += self.compute_R_s(test_data[query,:], rel_data[query,:], topk)
            
            return sum_r / test_data.shape[0]


    def eval_r1(self, test_data, rel_data):
        test_data.columns = ['Query', 'Item Code', 'Item Rank']
        rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']
        unique_queries = test_data['Query'].unique()
        sum_r = 0.0
        for query in unique_queries:
            query_test_data = test_data[test_data['Query'] == query]
            query_rel_data = rel_data[rel_data['Query'] == query]
            ra_list, rel_list = self.covert_pd_to_csv(query_test_data, query_rel_data)
            sum_r += self.compute_P_r(ra_list, rel_list, 1)
        
        return sum_r / len(unique_queries)
    
    
    def eval_r(self, test_data, rel_data, topk = 1):
        test_data.columns = ['Query', 'Item Code', 'Item Rank']
        rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']
        unique_queries = test_data['Query'].unique()
        sum_r = 0.0
        for query in unique_queries:
            query_test_data = test_data[test_data['Query'] == query]
            query_rel_data = rel_data[rel_data['Query'] == query]
            ra_list, rel_list = self.covert_pd_to_csv(query_test_data, query_rel_data)
            sum_r += self.compute_R_r(ra_list, rel_list, topk)
        
        return sum_r / len(unique_queries)



    def eval_p(self, test_data, rel_data, topk = None):
        test_data.columns = ['Query', 'Item Code', 'Item Rank']
        rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']
        unique_queries = test_data['Query'].unique()
        sum_p = 0.0
        for query in unique_queries:
            query_test_data = test_data[test_data['Query'] == query]
            query_rel_data = rel_data[rel_data['Query'] == query]
            ra_list, rel_list = self.covert_pd_to_csv(query_test_data, query_rel_data)
            if (topk is None):
                topk = np.sum(rel_list > 0)
            sum_p += self.compute_P_r(ra_list, rel_list, topk)

        return sum_p / len(unique_queries)
    """
    Compute ndcg

    compute_ndcg_r:
        list: 1 * item, 一维Numpy数组, 数组内存放排名rank(排名)
        rel_list: 1 * item, 一维Numpy数组, 数组内存放相关性分数
    """
    def compute_dcg(self, rank_list, rel_list, topk):
        dcg = 0.0

        if (topk > len(rel_list)):
            # print("Warning: Calculate NDCG metrics where topk is greater than the number of items")
            topk = len(rel_list)

        for i in range(topk):
            item_id = rank_list[i]
            rel = rel_list[item_id]
            dcg += (2 ** rel - 1) / math.log(i + 2, 2)

        return dcg

    def compute_ndcg_r(self, list, rel_list, topk):
        # 数组内存放item编号
        rank_list = np.argsort(list)
        rank_ideal_list = np.argsort(rel_list)[::-1]
        dcg = self.compute_dcg(rank_list, rel_list, topk)
        idcg = self.compute_dcg(rank_ideal_list, rel_list, topk)

        if (idcg == 0):
            # print("Warning: There is a situation where idcg is 0 in the evaluation")
            return 0
        else:
            return dcg / idcg
    
    """
    eval_ndcg:
        用于对算法输出的结果进行评测
        test_data: 
            - csv文件格式
            - Query | Item Code | Item Rank
        rel_data:
            - Query | 0 | Item | Relevance
    """
    def eval_ndcg(self, test_data, rel_data, topk = None):
        test_data.columns = ['Query', 'Item Code', 'Item Rank']
        rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']

        unique_queries = test_data['Query'].unique()
        sum_ndcg = 0.0
        for query in unique_queries:
            query_test_data = test_data[test_data['Query'] == query]

            query_rel_data = rel_data[rel_data['Query'] == query]
            ra_list, rel_list = self.covert_pd_to_csv(query_test_data, query_rel_data)
            if (topk is None):
                topk = np.sum(rel_list > 0)
            sum_ndcg += self.compute_ndcg_r(ra_list, rel_list, topk)
        
        return sum_ndcg / len(unique_queries)
    

if __name__ == '__main__':
    # RA_test_loc = r'C:\Users\2021\Desktop\Validate_SRA\Test_data\R_FLAGR_testdata_RRAexact.csv'
    # RA_rel_loc = r'C:\Users\2021\Desktop\Validate_SRA\Test_data\FLAGR_testdata_qrels.csv'
    # RA_test_data = pd.read_csv(RA_test_loc, header=None)
    # RA_rel_data = pd.read_csv(RA_rel_loc, header=None)
    # evalution = Evaluation()
    # for i in range (20, 100, 20):
    #     p = evalution.eval_p(RA_test_data, RA_rel_data, i)
    #     print("p@{0}:{1}".format(i, p))
    """
    Validate Weighted Borda
    """

    # sum_ndcg = 0.0
    # sum_p = 0.0
    # sum_map = 0.0
    # for i in range(1, 5 + 1):
    #     RA_test_loc = r'C:\Users\2021\Desktop\Validate_SRA\Validate_Weighted_Borda\result_MQ2008-agg\version_np\result_supBorda_MQ2008-agg-Fold' + str(i) + '.csv'
    #     RA_rel_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold' + str(i) + r'\rel_test.csv'
    #     evalution = Evaluation()
    #     RA_test_data = pd.read_csv(RA_test_loc, header=None)
    #     RA_rel_data = pd.read_csv(RA_rel_loc, header=None)
    #     sum_ndcg += evalution.eval_ndcg(RA_test_data, RA_rel_data, 10)
    #     sum_p += evalution.eval_p(RA_test_data, RA_rel_data, 10)
    #     sum_map += evalution.eval_map(RA_test_data, RA_rel_data,10)
    # print("ndcg@{0}:{1}".format(10, sum_ndcg / 5))
    # print("precision@{0}:{1}".format(10, sum_p / 5))
    # print("map:{0}".format(sum_map / 5))

    """
    Validate 
    """
    RA_test_loc = r'C:\Users\2021\Desktop\Supervised RA\test.csv'
    RA_rel_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold1\rel_test.csv'
    evalution = Evaluation()
    RA_test_data = pd.read_csv(RA_test_loc, header=None)
    RA_rel_data = pd.read_csv(RA_rel_loc, header=None)
    
    ndcg = evalution.eval_ndcg(RA_test_data, RA_rel_data, 10)
    print("ndcg@{0}:{1}".format(10, ndcg))

    p = evalution.eval_p(RA_test_data, RA_rel_data, 10)
    print("precision@{0}:{1}".format(10, p))

    map = evalution.eval_map(RA_test_data, RA_rel_data, 10)
    print("map:{0}".format(map))


    """
    Validate
    """
    # RA_test_loc = r'C:\Users\2021\Desktop\Supervised RA\test.csv'
    # RA_rel_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold1\rel_test.csv'
    # evalution = Evaluation()
    # RA_test_data = pd.read_csv(RA_test_loc, header=None)
    # RA_rel_data = pd.read_csv(RA_rel_loc, header=None)
    
    # ndcg = evalution.eval_ndcg(RA_test_data, RA_rel_data, 10)
    # print("ndcg@{0}:{1}".format(10, ndcg))

    # p = evalution.eval_p(RA_test_data, RA_rel_data, 10)
    # print("precision@{0}:{1}".format(10, p))

    # map = evalution.eval_map(RA_test_data, RA_rel_data, 10)
    # print("map:{0}".format(map))


    """
    Evolution
    """
    # RA_test_loc = r'C:\Users\2021\Desktop\Supervised RA\test.csv'
    # RA_rel_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold1\rel_test.csv'
    # evalution = Evaluation()
    # RA_test_data = pd.read_csv(RA_test_loc, header=None)
    # RA_rel_data = pd.read_csv(RA_rel_loc, header=None)
    
    
    