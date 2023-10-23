import pandas as pd
import numpy as np
import scipy.io
import inspect


# Import the PyFLAGR modules for rank aggregation
import pyflagr.Linear as Linear
import pyflagr.Majoritarian as Majoritarian
import pyflagr.MarkovChains as MarkovChains
import pyflagr.Kemeny as Kemeny
import pyflagr.RRA as RRA
import pyflagr.Weighted as Weighted
import inspect
# Code snippet for displaying dataframes side by side
from IPython.display import display_html
from itertools import chain, cycle


def display_side_by_side(*args, titles=cycle([''])):
    html_str = ''
    x = 1
    for df, title in zip(args, chain(titles, cycle(['</br>']))):
        html_str += '<th style="text-align:center"><td style="vertical-align:top">'
        html_str += f'<h2 style="text-align: center;">{title}</h2>'
        html_str += df.to_html().replace('table', 'table style="display:inline"')
        html_str += '</td></th>'
    display_html(html_str, raw=True)


def changeToMatlabModel(data):

    df_out = pd.DataFrame(data)

    # 提取所需的数据列
    query_column = df_out['Query'].values
    item_code_column = df_out['ItemCode'].values
    item_rank_column = df_out['ItemRank'].values

    # 获取唯一的Query和Item Code值以确定数组的大小
    unique_queries = np.unique(query_column)
    unique_item_codes = np.unique(item_code_column)

    print(len(unique_item_codes))

    # 创建一个二维数组，初始化为NaN
    matdata = np.empty((len(unique_queries), len(unique_item_codes)))
    matdata[:] = np.nan

    # 填充数组中的值
    for i, query in enumerate(unique_queries):
        for j, item_code in enumerate(unique_item_codes):
            mask = (query_column == query) & (item_code_column == item_code)
            if np.any(mask):
                item_rank = item_rank_column[mask][0]  # 取第一个匹配的Item Rank
                matdata[i, j] = item_rank
    return matdata


def RA_Condorcet():
    print(f"Running {inspect.currentframe().f_code.co_name}")
    condorcet = Majoritarian.CondorcetWinners(eval_pts=7)

    df_out, df_eval = condorcet.aggregate(input_file=lists, rels_file=qrels)

    df_out.to_csv('Transition_file.csv', index=True)
    new_df_out = pd.read_csv('Transition_file.csv')
    new_columns = ['Query', 'Dataset', 'ItemCode', 'ItemRank', 'ItemScore']
    new_df_out.columns = new_columns
    algo_name = inspect.currentframe().f_code.co_name
    outputcsvfile = 'rank-result-'+datasetname+'-'+algo_name + '.csv'
    new_df_out.to_csv(outputcsvfile, index=False, header=None)

#    print(new_df_out)
    # mat_df_out = changeToMatlabModel(new_df_out)

    # algo_name = inspect.currentframe().f_code.co_name
    # outputfile = 'rank-result-'+datasetname+'-'+algo_name + '.mat'
    # scipy.io.savemat(outputfile, {'result': mat_df_out})


def RA_Copeland():
    print(f"Running {inspect.currentframe().f_code.co_name}")
    copeland = Majoritarian.CopelandWinners(eval_pts=7)

    df_out, df_eval = copeland.aggregate(input_file=lists, rels_file=qrels)

    df_out.to_csv('Transition_file.csv', index=True)
    new_df_out = pd.read_csv('Transition_file.csv')
    new_columns = ['Query', 'Dataset', 'ItemCode', 'ItemRank', 'ItemScore']
    new_df_out.columns = new_columns

    algo_name = inspect.currentframe().f_code.co_name
    outputcsvfile = 'rank-result-'+datasetname+'-'+algo_name + '.csv'
    new_df_out.to_csv(outputcsvfile, index=False, header=None)

#    print(new_df_out)
    # mat_df_out = changeToMatlabModel(new_df_out)

    # algo_name = inspect.currentframe().f_code.co_name
    # outputfile = 'rank-result-'+datasetname+'-'+algo_name + '.mat'
    # scipy.io.savemat(outputfile, {'result': mat_df_out})


def RA_Outranking_Approach():
    print(f"Running {inspect.currentframe().f_code.co_name}")
    outrank = Majoritarian.OutrankingApproach(eval_pts=7)

    df_out, df_eval = outrank.aggregate(input_file=lists, rels_file=qrels)

    df_out.to_csv('Transition_file.csv', index=True)
    new_df_out = pd.read_csv('Transition_file.csv')
    new_columns = ['Query', 'Dataset', 'ItemCode', 'ItemRank', 'ItemScore']
    new_df_out.columns = new_columns

    algo_name = inspect.currentframe().f_code.co_name
    outputcsvfile = 'rank-result-'+datasetname+'-'+algo_name + '.csv'
    new_df_out.to_csv(outputcsvfile, index=False, header=None)


#    print(new_df_out)
    # mat_df_out = changeToMatlabModel(new_df_out)
    # algo_name = inspect.currentframe().f_code.co_name
    # outputfile = 'rank-result-'+datasetname+'-'+algo_name + '.mat'
    # scipy.io.savemat(outputfile, {'result': mat_df_out})


def RA_MC1():
    print(f"Running {inspect.currentframe().f_code.co_name}")
    mch = MarkovChains.MC1(eval_pts=7, max_iterations=50)

    df_out, df_eval = mch.aggregate(input_file=lists, rels_file=qrels)

    df_out.to_csv('Transition_file.csv', index=True)
    new_df_out = pd.read_csv('Transition_file.csv')
    new_columns = ['Query', 'Dataset', 'ItemCode', 'ItemRank', 'ItemScore']
    new_df_out.columns = new_columns

    algo_name = inspect.currentframe().f_code.co_name
    outputcsvfile = 'rank-result-'+datasetname+'-'+algo_name + '.csv'
    new_df_out.to_csv(outputcsvfile, index=False, header=None)

#    print(new_df_out)
    # mat_df_out = changeToMatlabModel(new_df_out)
    # algo_name = inspect.currentframe().f_code.co_name
    # outputfile = 'rank-result-'+datasetname+'-'+algo_name + '.mat'
    # scipy.io.savemat(outputfile, {'result': mat_df_out})


def RA_MC2():
    print(f"Running {inspect.currentframe().f_code.co_name}")
    mch = MarkovChains.MC2(eval_pts=7, max_iterations=50)

    df_out, df_eval = mch.aggregate(input_file=lists, rels_file=qrels)

    df_out.to_csv('Transition_file.csv', index=True)
    new_df_out = pd.read_csv('Transition_file.csv')
    new_columns = ['Query', 'Dataset', 'ItemCode', 'ItemRank', 'ItemScore']
    new_df_out.columns = new_columns

    algo_name = inspect.currentframe().f_code.co_name
    outputcsvfile = 'rank-result-'+datasetname+'-'+algo_name + '.csv'
    new_df_out.to_csv(outputcsvfile, index=False, header=None)

#    print(new_df_out)
    # mat_df_out = changeToMatlabModel(new_df_out)
    # algo_name = inspect.currentframe().f_code.co_name
    # outputfile = 'rank-result-'+datasetname+'-'+algo_name + '.mat'
    # scipy.io.savemat(outputfile, {'result': mat_df_out})


def RA_MC3():
    print(f"Running {inspect.currentframe().f_code.co_name}")
    mch = MarkovChains.MC3(eval_pts=7, max_iterations=50)

    df_out, df_eval = mch.aggregate(input_file=lists, rels_file=qrels)

    df_out.to_csv('Transition_file.csv', index=True)
    new_df_out = pd.read_csv('Transition_file.csv')
    new_columns = ['Query', 'Dataset', 'ItemCode', 'ItemRank', 'ItemScore']
    new_df_out.columns = new_columns

    algo_name = inspect.currentframe().f_code.co_name
    outputcsvfile = 'rank-result-'+datasetname+'-'+algo_name + '.csv'
    new_df_out.to_csv(outputcsvfile, index=False, header=None)

    # print(new_df_out)
    # mat_df_out = changeToMatlabModel(new_df_out)
    # algo_name = inspect.currentframe().f_code.co_name
    # outputfile = 'rank-result-'+datasetname+'-'+algo_name + '.mat'
    # scipy.io.savemat(outputfile, {'result': mat_df_out})


def RA_MC4():
    print(f"Running {inspect.currentframe().f_code.co_name}")
    mch = MarkovChains.MC4(eval_pts=7, max_iterations=50)

    df_out, df_eval = mch.aggregate(input_file=lists, rels_file=qrels)

    df_out.to_csv('Transition_file.csv', index=True)
    new_df_out = pd.read_csv('Transition_file.csv')
    new_columns = ['Query', 'Dataset', 'ItemCode', 'ItemRank', 'ItemScore']
    new_df_out.columns = new_columns

    algo_name = inspect.currentframe().f_code.co_name
    outputcsvfile = 'rank-result-'+datasetname+'-'+algo_name + '.csv'
    new_df_out.to_csv(outputcsvfile, index=False, header=None)

#    print(new_df_out)
    # mat_df_out = changeToMatlabModel(new_df_out)
    # algo_name = inspect.currentframe().f_code.co_name
    # outputfile = 'rank-result-'+datasetname+'-'+algo_name + '.mat'
    # scipy.io.savemat(outputfile, {'result': mat_df_out})


def RA_MCT():
    print(f"Running {inspect.currentframe().f_code.co_name}")
    mch = MarkovChains.MCT(eval_pts=7, max_iterations=50)

    df_out, df_eval = mch.aggregate(input_file=lists, rels_file=qrels)

    df_out.to_csv('Transition_file.csv', index=True)
    new_df_out = pd.read_csv('Transition_file.csv')
    new_columns = ['Query', 'Dataset', 'ItemCode', 'ItemRank', 'ItemScore']
    new_df_out.columns = new_columns

    algo_name = inspect.currentframe().f_code.co_name
    outputcsvfile = 'rank-result-'+datasetname+'-'+algo_name + '.csv'
    new_df_out.to_csv(outputcsvfile, index=False, header=None)

#    print(new_df_out)
    # mat_df_out = changeToMatlabModel(new_df_out)
    # algo_name = inspect.currentframe().f_code.co_name
    # outputfile = 'rank-result-'+datasetname+'-'+algo_name + '.mat'
    # scipy.io.savemat(outputfile, {'result': mat_df_out})


def RA_RRA_Exact():
    print(f"Running {inspect.currentframe().f_code.co_name}")
    robust = RRA.RRA(eval_pts=7, exact=True)

    df_out, df_eval = robust.aggregate(input_file=lists, rels_file=qrels)

    df_out.to_csv('Transition_file.csv', index=True)
    new_df_out = pd.read_csv('Transition_file.csv')
    new_columns = ['Query', 'Dataset', 'ItemCode', 'ItemRank', 'ItemScore']
    new_df_out.columns = new_columns

    algo_name = inspect.currentframe().f_code.co_name
    outputcsvfile = 'rank-result-'+datasetname+'-'+algo_name + '.csv'
    new_df_out.to_csv(outputcsvfile, index=False, header=None)

#    print(new_df_out)
    # mat_df_out = changeToMatlabModel(new_df_out)
    # algo_name = inspect.currentframe().f_code.co_name
    # outputfile = 'rank-result-'+datasetname+'-'+algo_name + '.mat'
    # scipy.io.savemat(outputfile, {'result': mat_df_out})


def RA_RRA():
    print(f"Running {inspect.currentframe().f_code.co_name}")
    robust = RRA.RRA(eval_pts=7, exact=False)

    df_out, df_eval = robust.aggregate(input_file=lists, rels_file=qrels)

    df_out.to_csv('Transition_file.csv', index=True)
    new_df_out = pd.read_csv('Transition_file.csv')
    new_columns = ['Query', 'Dataset', 'ItemCode', 'ItemRank', 'ItemScore']
    new_df_out.columns = new_columns

    algo_name = inspect.currentframe().f_code.co_name
    outputcsvfile = 'rank-result-'+datasetname+'-'+algo_name + '.csv'
    new_df_out.to_csv(outputcsvfile, index=False, header=None)

#    print(new_df_out)
    # mat_df_out = changeToMatlabModel(new_df_out)
    # algo_name = inspect.currentframe().f_code.co_name
    # outputfile = 'rank-result-'+datasetname+'-'+algo_name + '.mat'
    # scipy.io.savemat(outputfile, {'result': mat_df_out})


def RA_PrefRel():
    print(f"Running {inspect.currentframe().f_code.co_name}")
    prf_graph = Weighted.PreferenceRelationsGraph(
        alpha=0.1, beta=0.5, eval_pts=7)

    df_out, df_eval = prf_graph.aggregate(input_file=lists, rels_file=qrels)

    df_out.to_csv('Transition_file.csv', index=True)
    new_df_out = pd.read_csv('Transition_file.csv')
    new_columns = ['Query', 'Dataset', 'ItemCode', 'ItemRank', 'ItemScore']
    new_df_out.columns = new_columns

    algo_name = inspect.currentframe().f_code.co_name
    outputcsvfile = 'rank-result-'+datasetname+'-'+algo_name + '.csv'
    new_df_out.to_csv(outputcsvfile, index=False, header=None)

#    print(new_df_out)
    # mat_df_out = changeToMatlabModel(new_df_out)
    # algo_name = inspect.currentframe().f_code.co_name
    # outputfile = 'rank-result-'+datasetname+'-'+algo_name + '.mat'
    # scipy.io.savemat(outputfile, {'result': mat_df_out})


def RA_Agglomerative():
    print(f"Running {inspect.currentframe().f_code.co_name}")
    agg = Weighted.Agglomerative(c1=0.1, c2=0.2, eval_pts=7)

    df_out, df_eval = agg.aggregate(input_file=lists, rels_file=qrels)

    df_out.to_csv('Transition_file.csv', index=True)
    new_df_out = pd.read_csv('Transition_file.csv')
    new_columns = ['Query', 'Dataset', 'ItemCode', 'ItemRank', 'ItemScore']
    new_df_out.columns = new_columns

    algo_name = inspect.currentframe().f_code.co_name
    outputcsvfile = 'rank-result-'+datasetname+'-'+algo_name + '.csv'
    new_df_out.to_csv(outputcsvfile, index=False, header=None)

#    print(new_df_out)
    # mat_df_out = changeToMatlabModel(new_df_out)
    # algo_name = inspect.currentframe().f_code.co_name
    # outputfile = 'rank-result-'+datasetname+'-'+algo_name + '.mat'
    # scipy.io.savemat(outputfile, {'result': mat_df_out})


def RA_DIBRA():
    print(f"Running {inspect.currentframe().f_code.co_name}")
    method_1 = Weighted.DIBRA(aggregator='outrank', eval_pts=7)

    df_out, df_eval = method_1.aggregate(input_file=lists, rels_file=qrels)

    df_out.to_csv('Transition_file.csv', index=True)
    new_df_out = pd.read_csv('Transition_file.csv')
    new_columns = ['Query', 'Dataset', 'ItemCode', 'ItemRank', 'ItemScore']
    new_df_out.columns = new_columns

    algo_name = inspect.currentframe().f_code.co_name
    outputcsvfile = 'rank-result-'+datasetname+'-'+algo_name + '.csv'
    new_df_out.to_csv(outputcsvfile, index=False, header=None)

#    print(new_df_out)
    # mat_df_out = changeToMatlabModel(new_df_out)
    # algo_name = inspect.currentframe().f_code.co_name
    # outputfile = 'rank-result-'+datasetname+'-'+algo_name + '.mat'
    # scipy.io.savemat(outputfile, {'result': mat_df_out})


def RA_DIBRA_Prune():
    print(f"Running {inspect.currentframe().f_code.co_name}")
    method_2 = Weighted.DIBRA(eval_pts=7, gamma=1.5,
                              prune=True, d1=0.3, d2=0.05)

    df_out, df_eval = method_2.aggregate(input_file=lists, rels_file=qrels)

    df_out.to_csv('Transition_file.csv', index=True)
    new_df_out = pd.read_csv('Transition_file.csv')
    new_columns = ['Query', 'Dataset', 'ItemCode', 'ItemRank', 'ItemScore']
    new_df_out.columns = new_columns

    algo_name = inspect.currentframe().f_code.co_name
    outputcsvfile = 'rank-result-'+datasetname+'-'+algo_name + '.csv'
    new_df_out.to_csv(outputcsvfile, index=False, header=None)

#    print(new_df_out)
    # mat_df_out = changeToMatlabModel(new_df_out)
    # algo_name = inspect.currentframe().f_code.co_name
    # outputfile = 'rank-result-'+datasetname+'-'+algo_name + '.mat'
    # scipy.io.savemat(outputfile, {'result': mat_df_out})


lists = 'testdata.csv'
qrels = ''
datasetname = 'testdata'

RA_function_list = [RA_Copeland]

for func in RA_function_list:
    func()
