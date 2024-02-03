# Ranking Aggregation(RA) methods
More than 20 of unsupervised RA methods, 6 supervised RA methods and 1 semi-supervised RA methods were tested on our preprocessed dataset and the results of our preprocessing are in the folder. These methods including classic RA methods and state-of-the-art RA methods, if there is a need to test other datasets, please follow the comments in the code for dataset preprocessing and code modification.

## Get Started
To run your example in Python, open unsupervised RA methods_python/algorithm and then select the method you want to use. Please pay attention to change your dataset file to our corresponding format.

Define the input to the method as a csv file format, the columns of this CSV file must be organized in the following manner:

**Query, Voter name, Item Code, Item Rank**

where

- **Query** represents the topic for which the preference list is submitted,
- **Voter** is the name of the ranker who submitted a preference list for a particular **Query**,
- **Item Code** is a unique name that identifies each element of the preference lists,
- **Item Rank** is the preference rank assigned to an item by a Voter.

If you need to test our supervised or semi-supervised methods, then we need relevance judgments for the preference list elements of the primary input file for each query. It is organized in the following fashion:

**Query, 0, Item Code, Relevance**

where

- **Query** represents the topic for which the preference list is submitted,
- **0:** unused. This value must be always 0.
- **Item Code** is a unique name that identifies each element of the preference lists,
- **Relevance** is an integer value that represents the relevance of the item with respect to the mentioned Query. Typically, zero values represent irrelevant and incorrect elements and positive values represent relevant, correct and informative elements.

Similarly, we define the final output of the methods as a csv file which is organized in the following manner：**Query, Item Code, Item Rank**

We also provide a partially processed dataset in our dataset.zip file, you are welcome to use our code and test our code here!

## Test demonstrations
We selected several different unsupervised RA methods for simple test demonstrations, as can be seen in example1.ipynb (Dataset: MQ2007) and example2.ipynb (Dataset: Ice-cream flavor)

## Follow-up plan
We will be updating more RA methods for share use.

## Experiments

### Re-identification

In Re-identification(re-ID) datasets, we choose 6 feature extraction methods(BDB, BOT, Top-DB-Net-RK, LightMBN, FPB, LUPerson) to extract features from both query and gallery images, and then use the Euclidean method to combine the feature information of the combination of query and gallery to get the gallery scores under each query, and then eventually, for each query, we get the 6 basic rankings according to the scores in descending order. We evaluate our method on four image re-ID datasets(Market1501, DukeMTMC-reID and CUHK03 detected and labeled)

All experiments are conducted on Intel Xeon Silver 4215 (2.50GHz) and 4 Nvidia RTX A6000. It is important to note that MC1-4 methods are very difficult to test on the full Market1501 and DukeMTMC-reID datasets, requiring more than 40,000 hours in our experimental environment. Therefore, we conduct a cut-off operation for these two datasets on the basic rankings to refine our experiments as follows: we take out top-K items from all basic rankings to form a new itemset, and find the items in this itemset that were not originally present in specific basic ranking to add after the $k_{th}$ item of the basic ranking to finally obtain a new basic ranking. We use the MC1-4 method to aggregate the new basic rankings to a new one $R_{\tau}$. After aggregation, the items except itemset, we randomly sort them to the back of $R_{\tau}$ be the MC1-4 (top-K).

The result of initial rankings (BDB, BOT, Top-DB-Net-RK, LightMBN, FPB, LUPerson) is shown in Table 1.

<div align="center">
  
![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/7a8e9a23-d9dc-4262-a7a0-bc962fc3081b)

Table 1: Rank@1(%) and mAP(%) results for selected feature extraction methods on re-ID datasets.
</div>

we use official training sets to train basic re-ID and fully-supervised RA methods, and use official test sets to evaluate all RA methods.Table 2 presents the parameters of the semi-supervised and supervised methods, along with their type and the value that was set during the re-ID experiments. Note that a parameter setting of default means that for each query in the training set, the value taken is equal to the total number of relevant labels.
<div align="center">
  
![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/378eb7ed-bde0-434a-a76c-30fd245f2ff9)

Table 2: The parameters of the supervised ranking aggregation methods.
</div>

Table 3 shows the results of the experiment conducted on the four re-ID datasets, representing the quality of all ranking aggregation methods.

<div align="center">

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/f0c7f277-b2ba-4bce-8255-ed3f867db34b)

Table 3: Rank@1(%) and mAP(%) results for ranking aggregation methods on re-ID datasets.
</div>

### Recommendation system

In recommendation system dataset (MovieLens 1M), we perform a two-step pre-processing before giving them as input to the recommendation algorithms: (i) Binarization of the ratings of the datasets, as the items are considered only as relevant or irrelevant in the top-N recommendation task, whichs means an item is relevant to a user if its rating is greater than the median of the ratings given by the user. (ii) Removal of users and items that do not reach a predefined threshold value regarding frequency of ratings, which means we removed from the dataset infrequent items and users that rated very few items. Items rated by less than 10 users were removed, together with users that had rated less than 10 items.

On MovieLens 1M dataset,we divide the movies evaluated by each user into matrices $M_{train}$ and $M_{test}$. Specifically, we use 60% ratings of each user for $M_{train}$, and the remaining for $M_{test}$. Therefore, two matrices share the same set of users but have different movies recommended for these users. We use two criteria, Rank@1 and mAP@10 criteria, to evaluate the performance.

We selecte six recommendation algorithms(UserUser, BiasedMF, FunkSVD, ImplicitMF, ItemItem, MostPopular) in the experimental phase. All of them are available in the [LensKit](https://github.com/lenskit/lkpy) library. Table 4 presents the parameters of the recommendation algorithms, along with their type and the value that was set during our experiment. The names of the parameters used are consistent with the parameter names available in the LensKit library. We represent the quality of recommendations generated by six recommendation algorithms in Table 5.

<div align="center">

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/a5f07c8c-369b-4ce9-9e26-8e3ccdd268da)

Table 4: The parameters of the recommendation algorithms.

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/4124002a-1939-49f2-a6fc-146f84d2799a)

Table 5: Rank@1(%) and mAP(%) results for selected recommendation algorithms on MovieLens 1M dataset.
</div>

Different ranking aggregation methods will combine the six recommendations into a consensus one. We show their performance in Table 6.

<div align="center">

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/7b221691-5691-4b02-9cd4-268a65f7fcc1)

Table 6: Rank@1(%) and mAP(%) results for ranking aggregation methods on MovieLens 1M datasets
</div>

### Bioinformatics

In bioinformatics, we select a real dataset(NSCLC) related to cancer to conduct our experiment. Because there is no labeled data in the NSCLC dataset, we do not measure supervised and semi-supervised RA methods on NSCLC. The NSCLC dataset consists of four basic rankings which are of length 2270, 275, 543, 3501. The sources of them are [Kerkentzes et al. (2014)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4178426/), [Li et al. (2015)](https://pubmed.ncbi.nlm.nih.gov/26081616/), [Zhou et al. (2017)](https://www.nature.com/articles/onc2016242) and [Kim et al. (2013)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0055596). We consider the recall performance criteria in the aggregated list based on the top 400 and 800 genes. Thus, the result of all unsupervised RA methods is shown in Table 7.

<div align="center">

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/ff61aea3-a254-4c48-ada2-03c5dec653d8)

Table 7: Recall@400(%) and Recall@800(%) results for unsupervised RA methods on NSCLC datasets.
</div>

###  Social choices

We select five popular world university rankings: ARWU, QS, THE, US-NEW and URAP, where there is duplication of top-200 universities. In the five university rankings, because some universities appear in one ranking but not in another, we process the data for these five popular world university rankings. Specifically, we take the rank of the basic university ranked in its basic ranking for the duplicates. Furthermore, We first collect the set of all universities for these five universities rankings, and if a university does not appear in a particular basic ranking, we set that university to be the 201st in this basic ranking of that university ranking, and so on, until all five university rankings are processed. Eventually, we obtain and aggregate five ranking for an equal number of universities. We measure the normality and the overall impartiality to represent the quality of an aggregated list. The result of all unsupervised RA methods is shown in Table 8.

<div align="center">

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/f853bb7e-62b3-41c6-96b8-f57b156f973b)

Table 8: Normality and the overall of impartiality results for unsupervised RA methods on World University Ranking.
</div>

## Reference
 [[1]](https://openaccess.thecvf.com/content_ICCV_2019/html/Dai_Batch_DropBlock_Network_for_Person_Re-Identification_and_Beyond_ICCV_2019_paper.html) Dai, Zuozhuo, et al. "Batch dropblock network for person re-identification and beyond." Proceedings of the IEEE/CVF international conference on computer vision. 2019.
 
 [[2]](https://openaccess.thecvf.com/content_CVPRW_2019/html/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.html) Luo, Hao, et al. "Bag of tricks and a strong baseline for deep person re-identification." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops. 2019.

 [[3]](https://ieeexplore.ieee.org/abstract/document/9412017) Quispe, Rodolfo, and Helio Pedrini. "Top-db-net: Top dropblock for activation enhancement in person re-identification." 2020 25th International conference on pattern recognition (ICPR). IEEE, 2021.

 [[4]](https://ieeexplore.ieee.org/abstract/document/9506733) Herzog, Fabian, et al. "Lightweight multi-branch network for person re-identification." 2021 IEEE International Conference on Image Processing (ICIP). IEEE, 2021.

 [[5]](https://arxiv.org/abs/2108.01901) Zhang, Suofei, et al. "FPB: feature pyramid branch for person re-identification." arXiv preprint arXiv:2108.01901 (2021).

 [[6]](https://openaccess.thecvf.com/content/CVPR2021/html/Fu_Unsupervised_Pre-Training_for_Person_Re-Identification_CVPR_2021_paper.html) Fu, Dengpan, et al. "Unsupervised pre-training for person re-identification." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

 [[7]](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html) Zheng, Liang, et al. "Scalable person re-identification: A benchmark." Proceedings of the IEEE international conference on computer vision. 2015.

 [[8]](https://link.springer.com/chapter/10.1007/978-3-319-48881-3_2) Ristani, Ergys, et al. "Performance measures and a data set for multi-target, multi-camera tracking." European conference on computer vision. Cham: Springer International Publishing, 2016.

 [[9]](https://openaccess.thecvf.com/content_cvpr_2014/html/Li_DeepReID_Deep_Filter_2014_CVPR_paper.html) Li, Wei, et al. "Deepreid: Deep filter pairing neural network for person re-identification." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.

 [[10]](https://ieeexplore.ieee.org/abstract/document/9336268) Ye, Mang, et al. "Deep learning for person re-identification: A survey and outlook." IEEE transactions on pattern analysis and machine intelligence 44.6 (2021): 2872-2893.

 [[11]](https://dl.acm.org/doi/abs/10.1145/3365375) Oliveira, Samuel EL, et al. "Is rank aggregation effective in recommender systems? an experimental analysis." ACM Transactions on Intelligent Systems and Technology (TIST) 11.2 (2020): 1-26.

 [[12]](https://github.com/lenskit/lkpy) Michael D. Ekstrand. 2020. LensKit for Python: Next-Generation Software for Recommender Systems Experiments. In Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM '20). DOI:10.1145/3340531.3412778. arXiv:1809.03125 [cs.IR].
 
 [[13]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4178426/) Kerkentzes, Konstantinos, et al. "Hidden treasures in “ancient” microarrays: gene-expression portrays biology and potential resistance pathways of major lung cancer subtypes and normal tissue." Frontiers in oncology 4 (2014): 251.
 
 [[14]](https://link.springer.com/article/10.1007/s13277-015-3576-y) Li, Yafang, et al. "RNA-seq analysis of lung adenocarcinomas reveals different gene expression profiles between smoking and nonsmoking patients." Tumor Biology 36 (2015): 8993-9003.

 [[15]](https://www.nature.com/articles/onc2016242) Zhou, Y., et al. "microRNAs with AAGUGC seed motif constitute an integral part of an oncogenic signaling network." Oncogene 36.6 (2017): 731-745.

 [[16]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0055596) Kim, Sang Cheol, et al. "A high-dimensional, deep-sequencing study of lung adenocarcinoma in female never-smokers." PloS one 8.2 (2013): e55596.

 [[17]](https://academic.oup.com/bioinformatics/article/38/21/4927/6696211) Wang, Bo, et al. "Systematic comparison of ranking aggregation methods for gene lists in experimental results." Bioinformatics 38.21 (2022): 4927-4933.
