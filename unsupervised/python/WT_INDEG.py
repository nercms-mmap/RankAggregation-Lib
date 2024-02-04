# 参考文献：Preference relations based unsupervised rank aggregation for metasearch
# Tancilon：20231004
# 定义算法的顶层输入为csv文件格式，4列 Query | Voter name | Item Code | Item Rank
#      - Query 不要求是从1开始的连续整数
#      - Voter name 和 Item Code允许是字符串格式
# 定义算法的最终输出为csv文件格式：3列 Query | Item Code | Item Rank
#      - 注意输出的为排名信息，不是分数信息

# 数据输入可接受partial list，但不能处理并列排名的情况， Item Rank数值越小，排名越靠前
# 注意：该方法不适合社会选择领域
# 对于每一个查询算法的时间复杂度为：O(N * m^2) -- 其中N为输入排名数，m为总共的Item数量

# 验证：该代码与FLAGR库中的实现代码在MQ2007-agg数据集上的结果一致。
import numpy as np
import pandas as pd
import csv

def judge_alpha_disagree(input_list, rank_preference_graph, i, j, alpha, beta):
    N = rank_preference_graph.shape[1]
    num_voters = rank_preference_graph.shape[0]
    # i is better than j
    n0 = 0
    # j is better than i
    n1 = 0
    for k in range(num_voters):
        if (rank_preference_graph[k, j, i] == 1):
            n0 += 1
        elif (rank_preference_graph[k, i, j] == 1):
            n1 += 1
        elif (np.isnan(input_list[k, i]) and ~np.isnan(input_list[k, j])):
            n1 += 1
        elif (~np.isnan(input_list[k, i]) and np.isnan(input_list[k, j])):
            n0 += 1
        else:
            continue
    
    if (n0 + n1 < (beta * N) // 1):
        return False
    
    if (n0 < alpha * (n0 + n1)):
        return True
    
    return False
    
def aggregate_graph(weight_of_ranking, rank_preference_graph, i, j):
    num_voters = rank_preference_graph.shape[0]
    ans = 0
    for k in range(num_voters):
        if (~np.isnan(rank_preference_graph[k, i, j])):
            ans += weight_of_ranking[k] * rank_preference_graph[k, i, j]
    return ans
    

def WT_INDEG(input_list, alpha, beta):
    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]

    # 建立输入列表的偏好图
    rank_preference_graph = np.full((num_voters, num_items, num_items), np.nan)
    # 对每个输入排名分别建立偏好图
    for k in range(num_voters):
        for i in range (num_items):
            for j in range(i + 1, num_items):
                #只考虑出现在第k个排名的项目对（i,j)
                if (np.isnan(input_list[k,i]) or np.isnan(input_list[k,j])):
                    continue
                else:
                    # i 排在 j 的前面
                    if (input_list[k,j] > input_list[k,i]):
                        rank_preference_graph[k, j, i] = 1
                    else:
                        rank_preference_graph[k, i, j] = 1
    # Assigning quality scores to input rankings 将质量分数分配给输入排名
    rank_quality_score = np.zeros(num_voters)
    for i in range(num_items):
        for j in range(num_items):
            if (i == j):
                continue
            else:
                flag = judge_alpha_disagree(input_list, rank_preference_graph, i, j, alpha, beta)
                for k in range(num_voters):
                    if (flag == True and ((input_list[k,j] > input_list[k,i]) or (~np.isnan(input_list[k, i]) and np.isnan(input_list[k, j])))):
                        rank_quality_score[k] += 1
                    elif (np.isnan(input_list[k, i]) and np.isnan(input_list[k, j])):
                        rank_quality_score[k] += 0.5
                    else:
                        continue


    weight_of_ranking = np.zeros(num_voters)
    comb = num_items * (num_items - 1) / 2
    for k in range(num_voters):
        weight_of_ranking[k] = 1 - rank_quality_score[k] / comb
    
    weigthted_indgree = np.zeros(num_items)
    for j in range(num_items):
        for i in range(num_items):
            weigthted_indgree[j] += aggregate_graph(weight_of_ranking, rank_preference_graph, i, j)

    first_row = weigthted_indgree 
    # 进行排序并返回排序后的列索引
    sorted_indices = np.argsort(first_row)[::-1]
    
    currrent_rank = 1
    result = np.zeros(num_items)
    for index in sorted_indices:
        result[index] = currrent_rank
        currrent_rank += 1

    return result


def PreferenceRelationsGraph(input, output, alpha=0.5, beta=0.5):
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
        rank = WT_INDEG(input_list, alpha, beta)

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
