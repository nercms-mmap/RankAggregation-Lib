
# 参考文献：An Outranking Approach for Rank Aggregation in Information Retrieval
# Wsy：20231008
# 定义算法的顶层输入为csv文件格式，4列 Query | Voter name | Item Code | Item Rank
#      - Query 不要求是从1开始的连续整数
#      - Voter name 和 Item Code允许是字符串格式
# 定义算法的最终输出为csv文件格式：3列 Query | Item Code | Item Rank
#      - 注意输出的为排名信息，不是分数信息

# Item Rank数值越小，排名越靠前
# 四个超参数设置位于 [0, 1]：PREF_THRESHOLD, VETO_THRESHOLD, CONC_THRESHOLD, DISC_THRESHOLD 
# 数据输入可接受partial list 对于partial list 本代码的转化方法是将未排序的项目全部并列放在最后一名
# 验证：该代码与FLAGR库中的实现代码在模拟数据集上的结果基本一致。

import numpy as np
import pandas as pd
import csv
from preprocess import Map

def PartialToFull(input_list):
    # 扩充为full list的方式是将未排序的项目全部并列放在最后一名
    num_voters = input_list.shape[0]
    list_numofitems = np.zeros(num_voters)

    for k in range(num_voters):
        max_rank = np.nanmax(input_list[k])
        list_numofitems[k] = max_rank
        input_list[k] = np.nan_to_num(input_list[k], nan = max_rank + 1)

    return input_list, list_numofitems

def calculate_rank(outranking_matrix):
    result = []

    original_indices = list(range(outranking_matrix.shape[0]))

    while outranking_matrix.shape[0] > 0 and outranking_matrix.shape[1] > 0:
        row_sums = outranking_matrix.sum(axis=1)
        col_sums = outranking_matrix.sum(axis=0)

        scores = row_sums - col_sums

        max_score_index = np.argmax(scores)

        result.append(original_indices[max_score_index])

        original_indices.pop(max_score_index)

        outranking_matrix = np.delete(
            outranking_matrix, max_score_index, axis=0)
        outranking_matrix = np.delete(
            outranking_matrix, max_score_index, axis=1)

    return result


def OutrankingAgg(input_lists, list_numofitems, PREF_THRESHOLD, VETO_THRESHOLD, CONC_THRESHOLD, DISC_THRESHOLD):

    num_voter = input_lists.shape[0]
    num_item = input_lists.shape[1]
    print(num_item, num_voter)
    
    concordance_threshold = CONC_THRESHOLD * num_voter
    discordance_threshold = DISC_THRESHOLD * num_voter

    Outranking_maxtrix = np.zeros((num_item, num_item))

    for i in range(num_item):
        for j in range(num_item):
            if (i != j):
                concordance = 0
                discordance = 0
                for v in range(num_voter):
                    preference_threshold = PREF_THRESHOLD * list_numofitems[v]
                    veto_threshold = VETO_THRESHOLD * list_numofitems[v]

                    if (input_lists[v][i] <= input_lists[v][j]-preference_threshold):
                        concordance += 1
                    if (input_lists[v][i] >= input_lists[v][j]+veto_threshold):
                        discordance += 1
                if (concordance >= concordance_threshold and discordance <= discordance_threshold):
                    Outranking_maxtrix[i][j] = 1

    print(Outranking_maxtrix)
    item_ranked = calculate_rank(Outranking_maxtrix)
    return item_ranked


def Outranking(input_file_path, output_file_path, PREF_THRESHOLD, VETO_THRESHOLD, CONC_THRESHOLD, DISC_THRESHOLD, is_partial_list=True):
    df = pd.read_csv(input_file_path, header=None)
    df.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']

    unique_queries = df['Query'].unique()

    result = []

    for query in unique_queries:
        query_data = df[df['Query'] == query]
        int_to_item_map, int_to_voter_map, item_to_int_map, voter_to_int_map, input_lists = Map(
            query_data)
        

        if (is_partial_list == True):
            full_input_lists, list_numofitems = PartialToFull(input_lists)


        item_ranked = OutrankingAgg(
            full_input_lists,list_numofitems, PREF_THRESHOLD, VETO_THRESHOLD, CONC_THRESHOLD, DISC_THRESHOLD)
        print(item_ranked)
        for i in range(len(item_ranked)):
            item_code = int_to_item_map[item_ranked[i]]
            item_rank = i+1
            new_row = [query, item_code, item_rank]
            result.append(new_row)

    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in result:
            writer.writerow(row)
