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

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/893b5265-2e48-4187-b043-23e2f163b0ea)

Table 3: Rank@1(%) and mAP(%) results for ranking aggregation methods on re-ID datasets.
</div>

### Recommendation system

In recommendation system dataset (MovieLens 1M), we perform a two-step pre-processing before giving them as input to the recommendation algorithms: (i) Binarization of the ratings of the datasets, as the items are considered only as relevant or irrelevant in the top-N recommendation task, whichs means an item is relevant to a user if its rating is greater than the median of the ratings given by the user. (ii) Removal of users and items that do not reach a predefined threshold value regarding frequency of ratings, which means we removed from the dataset infrequent items and users that rated very few items. Items rated by less than 10 users were removed, together with users that had rated less than 10 items.

On MovielLens 1M dataset,we divide the movies evaluated by each user into matrices $M_{train}$ and $M_{test}$. Specifically, we use 60% ratings of each user for $M_{train}$, and the remaining for $M_{test}$. Therefore, two matrices share the same set of users but have different movies recommended for these users. We use two criteria, Rank@1 and mAP@10 criteria, to evaluate the performance.

We selecte six recommendation algorithms(UserUser, BiasedMF, FunkSVD, ImplicitMF, ItemItem, MostPopular) in the experimental phase. All of them are available in the [LensKit](https://github.com/lenskit/lkpy) library. Table 4 presents the parameters of the recommendation algorithms, along with their type and the value that was set during our experiment. The names of the parameters used are consistent with the parameter names available in the LensKit library. We represent the quality of recommendations generated by six recommendation algorithms in Table 5.

<div align="center">

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/a5f07c8c-369b-4ce9-9e26-8e3ccdd268da)

Table 4: The parameters of the recommendation algorithms.

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/b83a0e03-78a2-4e24-9c95-9450dedb0043)

Table 5: Rank@1 and mAP results for selected recommendation algorithms on MovielLens 1M dataset.
</div>

Different ranking aggregation methods will combine the six recommendations into a consensus one. We show their performance in Table 6.

<div align="center">

![1](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/f6a3655e-1f23-4726-883d-d67a82013523)

Table 6: Rank@1 and mAP results for ranking aggregation methods on MovielLens 1M datasets
</div>



