# unsupervised-RA-methods
These unsupervised RA methods were tested on our preprocessed dataset and the results of our preprocessing are in the folder. There are more than 20 of these methods including classic RA methods and state-of-the-aet RA methods, if there is a need to test other datasets, please follow the comments in the code for dataset preprocessing and code modification.

## Getting Started
To run your example in Python, open unsupervised RA methods_python/algorithm and then select the method you want to use. Please pay attention to change your dataset file to our corresponding format:  
Define the input to the method as a csv file format: Query | Voter name | Item Code | Item Rank  
Define the final output of the method as a csv file format： Query | Item Code | Item Rank  
We also provide a partially processed dataset in our dataset.zip file, you are welcome to use our code and test our code here!

## Test demonstrations
We selected several different unsupervised RA methods for simple test demonstrations, as can be seen in example1.ipynb (Dataset: MQ2007) and example2.ipynb (Dataset: Ice-cream flavor)

## Follow-up plan
We will be updating more RA methods for share use.

## Experiments

### re-ID 

In re-ID datasets, we choose 6 feature extraction methods to extract features from both query and gallery images (BDB [@1], BOT [@2], Top-DB-Net-RK [@3] , LightMBN [@4] , FPB [@5] , LUPerson [@6] ), and then use the Euclidean method to combine the feature information of the combination of query and gallery to get the gallery scores under each query, and then eventually, for each query, we get the 6 basic rankings according to the scores in descending order. We evaluate our method on four image re-ID datasets(Market1501, DukeMTMC-reID and CUHK03 detected and labeled)

All experiments are conducted on Intel Xeon Silver 4215 (2.50GHz) and 4 Nvidia RTX A6000. It is important to note that MC1-4 methods are very difficult to test on the full Market1501 and DukeMTMC-reID datasets, requiring more than 40,000 hours in our experimental environment. Therefore, we conduct a cut-off operation for these two datasets on the basic rankings to refine our experiments as follows: we take out top-K items from all basic rankings to form a new itemset, and find the items in this itemset that were not originally present in specific basic ranking to add after the $k_{th}$ item of the basic ranking to finally obtain a new basic ranking. We use the MC1-4 method to aggregate the new basic rankings to a new one $R_{\tau}$. After aggregation, the items except itemset, we randomly sort them to the back of $R_{\tau}$ be the MC1-4 (top-K).

The result of initial rankings (BDB, BOT, Top-DB-Net-RK, LightMBN, FPB, LUPerson) is shown in Table 1.

<div align="center">

|                 |         | Market1501 | Market1501 | DukeMTMC-ReID | DukeMTMC-ReID | CUHK03(detected) | CUHK03(detected) | CUHK03(labeled) | CUHK03(labeled) |
|:---------------:|:-------:|:----------:|:----------:|:-------------:|:-------------:|:----------------:|:----------------:|:---------------:|:---------------:|
| Initial Ranking |  Venue  |     R@1    |     mAP    |      R@1      |      mAP      |        R@1       |        mAP       |       R@1       |       mAP       |
|       BDB       | ICCV'19 |    94.21   |    85.08   |     88.11     |     74.83     |       76.07      |       71.47      |      78.36      |      74.62      |
|       BOT       | CVPR'19 |    95.31   |    93.7    |     90.26     |     88.34     |       69.93      |       71.58      |       72.5      |      74.59      |
|  Top-DB-Net+RK  | ICPR'21 |    95.52   |    93.97   |     90.48     |     87.77     |       85.79      |       86.26      |      87.86      |      88.43      |
|     LightMBN    | ICIP'21 |    96.41   |    91.25   |     92.01     |     83.28     |       85.93      |       81.53      |      87.21      |       83.9      |
|       FPB       | CVPR'21 |    95.46   |    89.98   |     89.72     |     81.88     |       81.57      |       78.03      |      84.29      |      81.28      |
|     LUPerson    | CVPR'21 |    96.26   |    90.73   |     90.26     |     81.74     |       70.93      |       67.42      |      74.64      |      72.53      |

Table 1
</div>

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/46c4cf0e-6982-4906-91d6-48903062dc99)



### Recommender system


## Reference
- [BDB](https://github.com/daizuozhuo/batch-dropblock-network) : Batch dropblock network for person re-identification and beyond

[2] 

[3] 
- - -
If you have problems using the code or have suggestions for changes, please contact waii2022@whu.edu.cn
