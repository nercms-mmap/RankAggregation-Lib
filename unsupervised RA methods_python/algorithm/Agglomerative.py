# 参考文献：A Weighted Rank aggregation approach towards crowd opinion analysis
# Tancilon：20231012
# 定义算法的顶层输入为csv文件格式，4列 Query | Voter name | Item Code | Item Rank
#      - Query 不要求是从1开始的连续整数
#      - Voter name 和 Item Code允许是字符串格式
# 定义算法的最终输出为csv文件格式：3列 Query | Item Code | Item Rank
#      - 注意输出的为排名信息，不是分数信息

# 数据输入可接受partial list 对于partial list 本代码的转化方法是将未排序的项目全部并列放在最后一名
# Item Rank数值越小，排名越靠前
# 注意：该方法可能不适合社会选择领域

import numpy as np
import pandas as pd
import csv


def PartialToFull(input_list):
    # 扩充为full list的方式是将未排序的项目全部并列放在最后一名
    num_voters = input_list.shape[0]

    for k in range(num_voters):
        max_rank = np.nanmax(input_list[k])
        input_list[k] = np.nan_to_num(input_list[k], nan = max_rank + 1)

    return input_list

# input_list_new: Voters*Items
def get_initial_weighted_ranks(input_list_new):
    num_voters = input_list_new.shape[0]
    num_items = input_list_new.shape[1]

    weighted_ranks = np.zeros(num_voters)

    for i in range(num_items):
        for j in range(i + 1, num_items):
            wins = 0
            losses = 0
            # 计算项目对（i，j)的输赢次数
            for k in range(num_voters):
                # i 排在 j 前面
                if (np.isnan(input_list_new[k, j]) and ~np.isnan(input_list_new[k, i])):
                    wins += 1
                if (input_list_new[k, i] < input_list_new[k, j]):
                    wins += 1
                # j 排在 i 前面
                if (np.isnan(input_list_new[k, i]) and ~np.isnan(input_list_new[k, j])):
                    losses += 1
                if (input_list_new[k, j] < input_list_new[k, i]):
                    losses += 1

            for k in range(num_voters):
                if (wins > losses and ((np.isnan(input_list_new[k, j]) and ~np.isnan(input_list_new[k, i])) or (input_list_new[k, i] < input_list_new[k, j]))):
                    weighted_ranks[k] += 1
                elif (wins < losses and ((np.isnan(input_list_new[k, i]) and ~np.isnan(input_list_new[k, j])) or (input_list_new[k, j] < input_list_new[k, i]))):
                    weighted_ranks[k] += 1

    return weighted_ranks

def get_spearman_rho(input_list_new, i, j):
    num_items = input_list_new.shape[1]

    denom = pow(num_items, 3) - num_items
    sum = 0

    for k in range(num_items):
        sum += (input_list_new[i, k] - input_list_new[j, k]) * (input_list_new[i, k] - input_list_new[j, k])
    
    rho = 1.0 - 6.0 * sum / denom
    return rho

def get_similarity_matrix(input_list_new):
    num_voters = input_list_new.shape[0]
    similarity_matrix = np.zeros((num_voters, num_voters))
    for i in range(num_voters):
        for j in range(i + 1, num_voters):
            
            sim = get_spearman_rho(input_list_new, i, j)
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    return similarity_matrix

def selected_rank(simlilarity_matrix):
    selected_rank_index1 = -1
    selected_rank_index2 = -1
    max_sim = float('-inf')
    num_voters = simlilarity_matrix.shape[0]

    for i in range(num_voters):
        for j in range(i + 1, num_voters):
            if (simlilarity_matrix[i, j] > max_sim):
                max_sim = simlilarity_matrix[i, j]
                selected_rank_index1 = i
                selected_rank_index2 = j
    return selected_rank_index1, selected_rank_index2


# input_list_new: Voters*Items
def Agglomerative_aggregation(input_list_new, c1, c2):
    num_voters = input_list_new.shape[0]
    num_items = input_list_new.shape[1]

    # 获得排名的初始权重
    weighted_ranks = get_initial_weighted_ranks(input_list_new)

    # 获得相似度矩阵，考虑排名的权重
    similarity_matrix = get_similarity_matrix(input_list_new)

    # 为初始排名计算位置得分
    input_list_score = np.zeros((num_voters, num_items))

    for k in range(num_voters):
        for i in range(num_items):
            input_list_score[k, i] = (num_items * num_items - 2 * num_items * (input_list_new[k, i] - 1) - num_items) / 2



    while (similarity_matrix.shape[0] > 1):
        # 选择两个最相似的排名
        selected_rank_index1, selected_rank_index2 = selected_rank(similarity_matrix)
        
        #合并两个排名并更新相关数据
        
        ## 计算合并后的分数
        MS = np.zeros(num_items)
        for i in range(num_items):
            # MS[i] = (weighted_ranks[selected_rank_index1] * input_list_score[selected_rank_index1, i] + weighted_ranks[selected_rank_index2] * input_list_score[selected_rank_index2, i]) / (weighted_ranks[selected_rank_index1] + weighted_ranks[selected_rank_index2])
            if np.isfinite(weighted_ranks[selected_rank_index1]) and np.isfinite(weighted_ranks[selected_rank_index2]) and (weighted_ranks[selected_rank_index1] + weighted_ranks[selected_rank_index2] != 0):
                MS[i] = (weighted_ranks[selected_rank_index1] * input_list_score[selected_rank_index1, i] + weighted_ranks[selected_rank_index2] * input_list_score[selected_rank_index2, i]) / (weighted_ranks[selected_rank_index1] + weighted_ranks[selected_rank_index2])
            else:
                MS[i] = 0  # 或者根据你的需求设置其他值

        
        ## 更新input_list_score
        input_list_score = np.delete(input_list_score, [selected_rank_index1, selected_rank_index2], axis=0)
        input_list_score = np.vstack((input_list_score, MS))
        ## 计算合并后的排名
        sorted_indices = np.argsort(MS)[::-1]
        currrent_rank = 1
        tmp_rank = np.zeros(num_items)
        for index in sorted_indices:
            tmp_rank[index] = currrent_rank
            currrent_rank += 1
    
        ## 更新inpu_list_new
        input_list_new = np.delete(input_list_new, [selected_rank_index1, selected_rank_index2], axis = 0)
        input_list_new = np.vstack((input_list_new, tmp_rank))

        ## 合并后更新相似度矩阵
        similarity_matrix = np.delete(similarity_matrix, [selected_rank_index1, selected_rank_index2], axis=0)  # 删除行
        similarity_matrix = np.delete(similarity_matrix, [selected_rank_index1, selected_rank_index2], axis=1)  # 删除列
        new_sim_row = np.zeros(similarity_matrix.shape[0]) # 计算合并后的相似度数值
        new_num_voters = input_list_new.shape[0]
        for k in range(similarity_matrix.shape[0]):
            new_sim_row[k] = get_spearman_rho(input_list_new, new_num_voters - 1, k)
        
        sum_sim = np.sum(new_sim_row)
        sim_weight = sum_sim / (new_num_voters)

        similarity_matrix = np.hstack((similarity_matrix, new_sim_row[:, np.newaxis]))
        new_sim_row = np.append(new_sim_row, 0)
        similarity_matrix = np.vstack((similarity_matrix, new_sim_row))

        ## 计算合并排名的新权重
        new_rank_weight = (c1 * (weighted_ranks[selected_rank_index1] + weighted_ranks[selected_rank_index2]) / 2 + c2 * sim_weight) / (c1 + c2)
        weighted_ranks = np.delete(weighted_ranks, [selected_rank_index1, selected_rank_index2])
        weighted_ranks = np.append(weighted_ranks, new_rank_weight)


    assert input_list_new.shape[0] == 1, "Aggregation result error!"

    return input_list_new[0]


def Agglomerative(input, output, c1=2.5, c2=1.5,is_partial_list=True):
    df = pd.read_csv(input,header=None)
    df.columns = ['Query','Voter Name', 'Item Code', 'Item Rank']

    # 获取唯一的Query值
    unique_queries = df['Query'].unique()
    # 创建一个空的DataFrame来存储结果
    result = []

    for query in unique_queries:
        # 筛选出当前Query的数据
        query_data = df[df['Query'] == query]

        # 创建空字典来保存Item Code和Voter Name的映射关系
        item_code_mapping = {}
        voter_name_mapping = {}

        # 获取唯一的Item Code和Voter Name值，并创建索引到整数的映射
        unique_item_codes = query_data['Item Code'].unique()
        unique_voter_names = query_data['Voter Name'].unique()

        # 建立整数到字符串的逆向映射
        item_code_reverse_mapping = {i: code for i, code in enumerate(unique_item_codes)}
        voter_name_reverse_mapping = {i: name for i, name in enumerate(unique_voter_names)}

        # 生成字符串到整数的映射
        item_code_mapping = {v: k for k, v in item_code_reverse_mapping.items()}
        voter_name_mapping = {v: k for k, v in voter_name_reverse_mapping.items()}

        # 创建Voter Name*Item Code的二维Numpy数组，初始值为0
        num_voters = len(unique_voter_names)
        num_items = len(unique_item_codes)
        #input_list = np.nan((num_voters, num_items))
        input_list = np.full((num_voters, num_items), np.nan)

        #填充数组
        for index, row in query_data.iterrows():
            voter_name = row['Voter Name']
            item_code = row['Item Code']
            item_rank = row['Item Rank']

            voter_index = voter_name_mapping[voter_name]
            item_index = item_code_mapping[item_code]

            input_list[voter_index, item_index] = item_rank
        
        # 如果是部分排序，先转化成完全排序
        if (is_partial_list == True):
            input_list_new = PartialToFull(input_list)
        else:
            input_list_new = input_list
            
        # 调用函数，获取排名信息
        rank = Agglomerative_aggregation(input_list_new, c1, c2)

        # 将结果添加到result_df中
        for item_code_index, item_rank in enumerate(rank):   
            item_code = item_code_reverse_mapping[item_code_index]
            #result_df = result_df.append({'Query': query, 'Item Code': item_code, 'Rank': item_rank}, ignore_index=True)
            new_row = [query, item_code, item_rank]
            result.append(new_row)
            
    with open(output, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in result:
            writer.writerow(row)