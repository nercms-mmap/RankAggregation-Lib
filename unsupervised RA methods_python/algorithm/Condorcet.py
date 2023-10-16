# Wsy, Tancilon 20231005
# fsw, evaluation 20231015
# 定义算法的顶层输入为csv文件格式，4列 Query | Voter name | Item Code | Item Rank
#      - Query 不要求是从1开始的连续整数
#      - Voter name 和 Item Code允许是字符串格式
# 定义算法的最终输出为csv文件格式：3列 Query | Item Code | Item Rank
#      - 注意输出的为排名信息，不是分数信息
# 输入接受partial list

# 正确性验证：本算法与FALGR库中的算法实现在MQ2007-agg数据集上结果一致。与matlab Condorcet.mat结果一致
import numpy as np
import pandas as pd
import csv

def win(input_list, i, j):
    num_voters = input_list.shape[0]
    win_i = 0
    win_j = 0
    for k in range(num_voters):
        if (np.isnan(input_list[k, i]) and np.isnan(input_list[k, j])):
            continue
        elif (np.isnan(input_list[k, i]) and ~np.isnan(input_list[k, j])):
            win_j += 1
        elif (~np.isnan(input_list[k, i]) and np.isnan(input_list[k, j])):
            win_i += 1
        elif (input_list[k, i] < input_list[k, j]):
            win_i += 1
        else:
            win_j += 1
    
    if (win_i > win_j):
        return 1
    
    elif (win_i < win_j):
        return -1
    else:
        return 0


def CondorcetAgg(input_list):
    num_items = input_list.shape[1]
    item_win_count = np.zeros(num_items)

    for i in range(num_items):
        for j in range(i + 1, num_items):
            # 项目对(i, j)中 i 赢了
            flag = win(input_list, i, j)
            if (flag == 1):
                item_win_count[i] += 1
            # j 赢了
            elif (flag == -1):
                item_win_count[j] += 1
            # 平局
            else:
                continue
    
    first_row = item_win_count
    # 进行排序并返回排序后的列索引
    sorted_indices = np.argsort(first_row)[::-1]
    
    currrent_rank = 1
    result = np.zeros(num_items)
    for index in sorted_indices:
        result[index] = currrent_rank
        currrent_rank += 1

    return result


def Condorcet(input, output):
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

        # 调用函数，获取排名信息
        rank = CondorcetAgg(input_list)

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