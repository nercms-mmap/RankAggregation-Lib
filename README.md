# Unsupervised-RA-methods
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

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/084ef917-f1dc-4c4c-ace3-8e1e61dfade4)

Table 5: Rank@1(%) and mAP(%) results for selected recommendation algorithms on MovieLens 1M dataset.
</div>

Different ranking aggregation methods will combine the six recommendations into a consensus one. We show their performance in Table 6.

<div align="center">

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/d44c073c-af6e-4755-8b12-22724e6951bf)

Table 6: Rank@1(%) and mAP(%) results for ranking aggregation methods on MovieLens 1M datasets
</div>

### Bioinformatics

In bioinformatics, we select a real dataset(NSCLC) related to cancer to conduct our experiment. Because there is no labeled data in the NSCLC dataset, we do not measure supervised and semi-supervised RA methods on NSCLC. The NSCLC dataset consists of four basic rankings which are of length 2270, 275, 543, 3501. The sources of them are [Kerkentzes et al. (2014)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4178426/), [Li et al. (2015)](https://pubmed.ncbi.nlm.nih.gov/26081616/), [Zhou et al. (2017)](https://www.nature.com/articles/onc2016242) and [Kim et al. (2013)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0055596). We consider the recall performance criteria in the aggregated list based on the top 400 and 800 genes. Thus, the result of all unsupervised RA methods is shown in Table 7.

<div align="center">

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/bff74db5-f28b-4161-9e38-53ea73ebed55)

Table 7: Recall@400(%) and Recall@800(%) results for unsupervised RA methods on NSCLC datasets.
</div>

###  Social choices

We select five popular world university rankings: ARWU, QS, THE, US-NEW and URAP, where there is duplication of top-200 universities. In the five university rankings, because some universities appear in one ranking but not in another, we process the data for these five popular world university rankings. Specifically, we take the rank of the basic university ranked in its basic ranking for the duplicates. Furthermore, We first collect the set of all universities for these five universities rankings, and if a university does not appear in a particular basic ranking, we set that university to be the 201st in this basic ranking of that university ranking, and so on, until all five university rankings are processed. Eventually, we obtain and aggregate five ranking for an equal number of universities. We measure the normality and the overall impartiality to represent the quality of an aggregated list. The result of all unsupervised RA methods is shown in Table 8.

<div align="center">

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/d5471b42-cfb7-4598-9c55-7ce6491b660d)

Table 8: Normality and the overall of impartiality results for unsupervised RA methods on World University Ranking.
</div>

