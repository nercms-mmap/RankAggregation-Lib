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

### Re-identification

In Re-identification(re-ID) datasets, we choose 6 feature extraction methods to extract features from both query and gallery images (BDB, BOT, Top-DB-Net-RK, LightMBN, FPB, LUPerson), and then use the Euclidean method to combine the feature information of the combination of query and gallery to get the gallery scores under each query, and then eventually, for each query, we get the 6 basic rankings according to the scores in descending order. We evaluate our method on four image re-ID datasets(Market1501, DukeMTMC-reID and CUHK03 detected and labeled)

All experiments are conducted on Intel Xeon Silver 4215 (2.50GHz) and 4 Nvidia RTX A6000. It is important to note that MC1-4 methods are very difficult to test on the full Market1501 and DukeMTMC-reID datasets, requiring more than 40,000 hours in our experimental environment. Therefore, we conduct a cut-off operation for these two datasets on the basic rankings to refine our experiments as follows: we take out top-K items from all basic rankings to form a new itemset, and find the items in this itemset that were not originally present in specific basic ranking to add after the $k_{th}$ item of the basic ranking to finally obtain a new basic ranking. We use the MC1-4 method to aggregate the new basic rankings to a new one $R_{\tau}$. After aggregation, the items except itemset, we randomly sort them to the back of $R_{\tau}$ be the MC1-4 (top-K).

The result of initial rankings (BDB, BOT, Top-DB-Net-RK, LightMBN, FPB, LUPerson) is shown in Table 1.

<div align="center">
  
![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/5c8e8139-5c1b-404d-87e4-b55bbe91d799)
Table 1: Rank@1(%) and mAP(%) results for selected feature extraction methods on re-ID datasets
</div>

we use official training sets to train basic re-ID and fully-supervised RA methods, and use official test sets to evaluate all RA methods.Table 2 presents the parameters of the semi-supervised and supervised methods, along with their type, and the value that was set during the re-ID experiments. Note that a parameter setting of default means that for each query in the training set, the value taken is equal to the total number of relevant labels.
<div align="center">
  
![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/0095cd44-5a2a-45df-8683-5afb506727b8)



Table 2: The parameters of the supervised ranking aggregation methods
</div>

The final result of all RA methods is shown in Table 3.

<div align="center">

![QQ截图20240131000101](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/4bf6c2e2-8187-4527-a48d-1d4dfd929304)


Table 3: Rank@1(%) and mAP(%) results for ranking aggregation methods on re-ID datasets
</div>






