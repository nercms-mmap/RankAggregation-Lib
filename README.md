# Rank Aggregation (RA) methods
20 unsupervised RA methods, 6 supervised RA methods and 1 semi-supervised RA methods were tested on our preprocessed datasets. These datasets cover the areas of person re-identification (re-ID), recommendation system, bioinformatics and social choices. The tested methods include both classical and state-of-the-art RA methods. If there is a need to test other datasets, please follow the instructions in the code comments for dataset preprocessing and necessary code modifications.

<table align="center">
    <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Unsupervised</b>
      </td>
      <td>
        <b>Supervised</b>
      </td>
      <td>
        <b>Semi-supervised</b>
      </td>
    </tr>
    <tr valign="top">
        <td>
            <ul>
                <li>$\textrm{CombMIN}$ <a href="#Comb">[1]</li>
                <li>$\textrm{CombMAX}$ <a href="#Comb">[1]</li>
                <li>$\textrm{CombSUM}$ <a href="#Comb">[1]</li>
                <li>$\textrm{CombANZ}$ <a href="#Comb">[1]</li>
                <li>$\textrm{CombMNZ}$ <a href="#Comb">[1]</li>
                <li>$\textrm{MC1}$ <a href="#MC">[2]</li>
                <li>$\textrm{MC2}$ <a href="#MC">[2]</li>
                <li>$\textrm{MC3}$ <a href="#MC">[2]</li>
                <li>$\textrm{MC4}$ <a href="#MC">[2]</li>
                <li>$\textrm{Borda count}$ <a href="#Borda">[3]</li>
                <li>$\textrm{Dowdall}$ <a href="#Dowdall">[4]</li>
                <li>$\textrm{Median}$ <a href="#Median">[5]</li>
                <li>$\textrm{RRF}$ <a href="#RRF">[6]</li>
                <li>$\textrm{iRANK}$ <a href="#iRank">[7]</li>
                <li>$\textrm{Mean}$ <a href="#Mean">[8]</li>
                <li>$\textrm{HPA}$ <a href="#HPA&postNDCG">[9]</li>
                <li>$\textrm{PostNDCG}$ <a href="#HPA&postNDCG">[9]</li>
                <li>$\textrm{ER}$ <a href="#ER">[10]</li>
                <li>$\textrm{CG}$ <a href="#CG">[11]</li>
                <li>$\textrm{DIBRA}$ <a href="#DIBRA">[12]</li>
        </td>
        <td>
            <ul>
                <li>$\textrm{wBorda}$ <a href="#wBorda">[13]</li>
                <li>$\textrm{CRF}$ <a href="#CRF">[14]</li>
                <li>$\textrm{CSRA}$ <a href="#CSRA">[15]</li>
                <li>$\textrm{AggRankDE}$ <a href="#AggRankDe">[16]</li>
                <li>$\textrm{IRA}_\textrm{R}$ <a href="#IRA">[17]</li>
                <li>$\textrm{IRA}_\textrm{S}$ <a href="#IRA">[17]</li>
                <li>$\textrm{QI-IRA}$ <a href="#QIIRA">[18]</li>
        </td>
        <td>
            <ul>
                <li>$\textrm{SSRA}$ <a href="#semi">[19]</li>
        </td>
    </tbody>
</table>

# Directory structure
```
│  README.md
│  
├─datasets
│  ├─FLAGR
│  ├─ice-cream
│  ├─MovieLens 1M
│  ├─MQ2008-agg
│  ├─NSCLC
│  ├─Re-ID
│  └─World University Ranking 2022
│      
└─unsupervised
│   ├─matlab
│   │      BordaCount.m
│   │      CG.m
│   │      CombANZ.m
│   │      CombMAX.m
│   │      CombMED.m
│   │      CombMIN.m
│   │      CombMNZ.m
│   │      CombSUM.m
│   │      Condorcet.m
│   │      DIBRA.m
│   │      Dowdall.m
│   │      EnsembleRanking.m
│   │      ER.m
│   │      HPA.m
│   │      hpa_func.m
│   │      ice-cream.mat
│   │      iRank.m
│   │      Matrix-ice-cream.mat
│   │      Mean.m
│   │      Median.m
│   │      PostNDCG.m
│   │      RRF.m
│   │      unsupervised RA methods.ipynb
│   │      
│   └─python
│           BordaCount.py
│           CG.py
│           CombANZ.py
│           CombMAX.py
│           CombMED.py
│           CombMIN.py
│           CombMNZ.py
│           CombSUM.py
│           Comb_Family.py
│           Dowdall.py
│           evaluate.py
│           MarkovChain.py
│           Mean.py
│           Medium.py
│           preprocess.py
│           RRF.py
│           run_algorithm.py
│           scorefunc.py
│           unsupervised RA methods.ipynb
│
├─supervised
│  ├─matlab
│  │  │  compute_AP.m
│  │  │  evaluation.m
│  │  │  m_QT_IRA.m
│  │  │  m_Rank_based_IRA.m
│  │  │  m_Score_based_IRA.m
│  │  │  QT_IRA.m
│  │  │  Rank_based_IRA.m
│  │  │  Score_based_IRA.m
│  │  │  
│  │  └─CSRA
│  │          
│  └─python
│          AggRankDE.py
│          CRF.py
│          Evaluation.py
│          WeightedBorda.py
│          
├─semi-supervised
│      SSRA.py
```

## Get Started
To run your example in Python, open the `unsupervised RA methods_python/algorithm` directory and then select the method you want to use. Please ensure you modify your dataset file to match our specified format.

Define the input to the method as a CSV file format. The columns in this CSV file should be organized as follows:

**Query, Voter, Item Code, Item Rank**

where

- **Query** is the topic for which the preference list is submitted.
- **Voter** is the name of the ranker who submitted a preference list for a particular **Query**.
- **Item Code** is a unique identifier that identifies each element of the preference lists.
- **Item Rank** is the preference rank assigned to an item by a Voter.

If you need to test our supervised or semi-supervised methods, then relevance judgments are required for the elements of the preference list in the primary input file for each query. It is organized as follows:

**Query, 0, Item Code, Relevance**

where

- **Query** is the topic for which the preference list is submitted.
- **0:** unused. This value must be always 0.
- **Item Code** is a unique identifier that identifies each element of the preference lists.
- **Relevance** is an integer value that represents the relevance of the item with respect to the mentioned Query. Typically, zero values represent irrelevant and incorrect elements and positive values represent relevant, correct and informative elements.

Similarly, we define the final output of the methods as a CSV file which is organized in the following manner：**Query, Item Code, Item Rank**.

We also provide a partially processed dataset in our `dataset.zip` file. You are welcome to use our code and test it here!

## Follow-up Plan
We will be updating and adding more RA methods for shared use.

## Experiments

### Re-identification
In Re-identification (re-ID) datasets, we choose 6 feature extraction methods (BDB [[20]](#BDB), BOT [[21]](#BOT), Top-DB-Net-RK [[22]](#top), LightMBN [[23]](#light), FPB [[24]](#FPB), LUPerson [[25]](#lu)) to extract features from both query and gallery images, and then use the Euclidean method to combine the feature information of the combination of query and gallery to get the gallery scores under each query, and then eventually, for each query, we get the 6 basic rankings according to the scores in descending order. We evaluate our method on four image re-ID datasets (Market1501 [[26]](#market), DukeMTMC-reID [[27]](#duke) and CUHK03 detected and labeled [[28]](#cuhk))

All experiments are conducted on Intel Xeon Silver 4215 (2.50GHz) and 4 Nvidia RTX A6000. It is important to note that MC1-4 methods are very difficult to test on the full Market1501 and DukeMTMC-reID datasets, requiring more than 40,000 hours in our experimental environment. Therefore, we conduct a cut-off operation for these two datasets on the basic rankings to refine our experiments as follows: we take out top-K items from all basic rankings to form a new itemset, and find the items in this itemset that were not originally present in specific basic ranking to add after the $k_{th}$ item of the basic ranking to finally obtain a new basic ranking. We use the MC1-4 method to aggregate the new basic rankings to a new one $R_{\tau}$. After aggregation, the items except itemset, we randomly sort them to the back of $R_{\tau}$ be the MC1-4 (top-K).

The result of basic rankings (BDB, BOT, Top-DB-Net-RK, LightMBN, FPB, LUPerson) is shown in Table 1.

<div align="center">
  
![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/bac6aec4-37eb-4a64-8915-c54238c561f3)

Table 1: Rank@1 (%) and mAP (%) [[29]](#rank1) results for selected feature extraction methods on re-ID datasets.
</div>

we use official training sets to train basic re-ID and fully-supervised RA methods, and use official test sets to evaluate all RA methods.Table 2 presents the parameters of the semi-supervised and supervised methods, along with their type and the value that was set during the re-ID experiments. Note that a parameter setting of default means that for each query in the training set, the value taken is equal to the total number of relevant labels.
<div align="center">
  
![SupPara](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/e65cbb92-416e-454e-af09-9a3eea982c7e)

Table 2: The parameters of the supervised rank aggregation methods.
</div>

Table 3 shows the results of the experiment conducted on the four re-ID datasets, representing the quality of all rank aggregation methods.

<div align="center">

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/930d9e18-3327-4ec7-ba95-0502672c3d40)

Table 3: Rank@1 (%) and mAP (%) [[30]](#RS) results for rank aggregation methods on re-ID datasets.
</div>

### Recommendation System

In recommendation system dataset (MovieLens 1M), we perform a two-step pre-processing before giving them as input to the recommendation algorithms: (i) Binarization of the ratings of the datasets, as the items are considered only as relevant or irrelevant in the top-N recommendation task, whichs means an item is relevant to a user if its rating is greater than the median of the ratings given by the user. (ii) Removal of users and items that do not reach a predefined threshold value regarding frequency of ratings, which means we removed from the dataset infrequent items and users that rated very few items. Items rated by less than 10 users were removed, together with users that had rated less than 10 items [[30]](#RS).

On MovieLens 1M dataset,we divide the movies evaluated by each user into matrices $M_{train}$ and $M_{test}$. Specifically, we use 60% ratings of each user for $M_{train}$, and the remaining for $M_{test}$. Therefore, two matrices share the same set of users but have different movies recommended for these users. We use two criteria, Rank@1 and mAP@10 criteria, to evaluate the performance.

We select six recommendation algorithms [[31]](#lenskit) (UserUser, BiasedMF, FunkSVD, ImplicitMF, ItemItem, MostPopular) in the experimental phase. All of them are available in the [LensKit](https://github.com/lenskit/lkpy) library. Table 4 presents the parameters of the recommendation algorithms, along with their type and the value that was set during our experiment. The names of the parameters used are consistent with the parameter names available in the LensKit library. We represent the quality of recommendations generated by six recommendation algorithms in Table 5.

<div align="center">

![RecPara](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/606baf2e-ef0a-4a1a-9f62-aa4d45b1a0e3)

Table 4: The parameters of the recommendation algorithms.

![Rec-ini](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/41b3c3ac-d53b-45f4-a74b-0272fd6f46a3)

Table 5: Rank@1 (%) and mAP (%) results for selected recommendation algorithms on MovieLens 1M dataset.
</div>

Different rank aggregation methods will combine the six recommendations into a consensus one. We show their performance in Table 6.

<div align="center">

![ml](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/a22ac931-c3b1-4412-9efd-414b2596fbcf)

Table 6: Rank@1 (%) and mAP@10 (%) [[30]](#RS) results for rank aggregation methods on MovieLens 1M datasets
</div>

### Bioinformatics

In bioinformatics, we select a real dataset (NSCLC) related to cancer to conduct our experiment. Because there is no labeled data in the NSCLC dataset, we do not measure supervised and semi-supervised RA methods on NSCLC. The NSCLC dataset consists of four basic rankings which are of length 2270, 275, 543, 3501. The sources of them are [[32]](#Kerkentzes), [[33]](#Li), [[34]](#Zhou) and [[35]](#Kim). We consider the recall performance criteria in the aggregated list based on the top 400 and 800 genes. Thus, the result of all unsupervised RA methods is shown in Table 7.

<div align="center">

![NSCLC](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/7dd85d0c-668d-4065-99cb-ac420d7cd8aa)

Table 7: Recall@400 (%) and Recall@800 (%) [[36]](#recall) results for unsupervised RA methods on NSCLC datasets.
</div>

###  Social Choices

We select five popular world university rankings: [[ARWU]](https://www.shanghairanking.com/), [[QS]](https://www.qs.com/rankings-performance/), [[THE]](https://www.int-res.com/abstracts/esep/v13/n2/p125-130/), [[US-NEW]](https://www.usnews.com/best-colleges)  and [[URAP]](https://urapcenter.org/), where there is duplication of top-200 universities. In the five university rankings, because some universities appear in one ranking but not in another, we process the data for these five popular world university rankings. Specifically, we take the rank of the basic university ranked in its basic ranking for the duplicates. Furthermore, We first collect the set of all universities for these five university rankings, and if a university that belongs to the set of all universities does not appear in a particular basic ranking, we set this university to be the 201st of the basic ranking, and so on, until all five university rankings are processed. Eventually, we obtain and aggregate five rankings for an equal number of universities. We measure the normality and the overall impartiality [[37]](#IEIR) to represent the quality of a consensus ranking. The result of all unsupervised RA methods is shown in Table 8.

<div align="center">

![SC](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/cc1d411b-2979-4c4c-94c1-b26faadf5a13)

Table 8: Normality and the overall of impartiality results for unsupervised RA methods on World University Ranking.
</div>

## References
<a id="Comb">[[1]](https://books.google.com.tw/books?hl=zh-CN&lr=&id=W8MZAQAAIAAJ&oi=fnd&pg=PA243&dq=Combination+of+multiple+searches.&ots=3XwVWFAQ5n&sig=EGO4Nkeo5BIsfg0HOpiHsnNPjm4&redir_esc=y#v=onepage&q=Combination%20of%20multiple%20searches.&f=false) Fox, Edward, and Joseph Shaw. "Combination of multiple searches." NIST special publication SP (1994): 243-243.</a>

<a id="MC">[[2]](https://dl.acm.org/doi/abs/10.1145/371920.372165) Dwork, Cynthia, et al. "Rank aggregation methods for the web." Proceedings of the 10th international conference on World Wide Web. 2001.</a>

<a id="Borda">[[3]](https://dl.acm.org/doi/abs/10.1145/383952.384007) Aslam, Javed A., and Mark Montague. "Models for metasearch." Proceedings of the 24th annual international ACM SIGIR conference on Research and development in information retrieval. 2001.</a>

<a id="Dowdall">[[4]](https://journals.sagepub.com/doi/abs/10.1177/0192512102023004002) Reilly, Benjamin. "Social choice in the south seas: Electoral innovation and the borda count in the pacific island countries." International Political Science Review 23.4 (2002): 355-372.</a>

<a id="Median">[[5]](https://dl.acm.org/doi/abs/10.1145/872757.872795) Fagin, Ronald, Ravi Kumar, and Dandapani Sivakumar. "Efficient similarity search and classification via rank aggregation." Proceedings of the 2003 ACM SIGMOD international conference on Management of data. 2003.</a>

<a id="RRF">[[6]](https://dl.acm.org/doi/abs/10.1145/1571941.1572114) Cormack, Gordon V., Charles LA Clarke, and Stefan Buettcher. "Reciprocal rank fusion outperforms condorcet and individual rank learning methods." Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval. 2009.</a>

<a id="iRANK">[[7]](https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/asi.21296) Wei, Furu, Wenjie Li, and Shixia Liu. "iRANK: A rank‐learn‐combine framework for unsupervised ensemble ranking." Journal of the American Society for Information Science and Technology 61.6 (2010): 1232-1243. 2009.</a>

<a id="Mean">[[8]](https://proceedings.mlr.press/v14/burges11a/burges11a.pdf) Burges, Christopher, et al. "Learning to rank using an ensemble of lambda-gradient models." Proceedings of the learning to rank Challenge. PMLR, 2011.</a>

<a id="HPA&postNDCG">[[9]](https://link.springer.com/chapter/10.1007/978-3-030-45442-5_17) Fujita, Soichiro, Hayato Kobayashi, and Manabu Okumura. "Unsupervised Ensemble of Ranking Models for News Comments Using Pseudo Answers." Advances in Information Retrieval: 42nd European Conference on IR Research, ECIR 2020, Lisbon, Portugal, April 14–17, 2020, Proceedings, Part II 42. Springer International Publishing, 2020.</a>

<a id="ER">[[10]](https://www.sciencedirect.com/science/article/pii/S0305048319308448) Mohammadi, Majid, and Jafar Rezaei. "Ensemble ranking: Aggregation of rankings produced by different multi-criteria decision-making methods." Omega 96 (2020): 102254.</a>

<a id="CG">[[11]](https://www.tandfonline.com/doi/abs/10.1080/01605682.2019.1657365) Xiao, Yu, et al. "Graph-based rank aggregation method for high-dimensional and partial rankings." Journal of the Operational Research Society 72.1 (2021): 227-236.</a>

<a id="DIBRA">[[12]](https://www.sciencedirect.com/science/article/abs/pii/S0957417422007710) Akritidis, Leonidas, et al. "An unsupervised distance-based model for weighted rank aggregation with list pruning." Expert Systems with Applications 202 (2022): 117435.</a>

<a id="wBorda">[[13]](https://ieeexplore.ieee.org/abstract/document/6495123) Pujari, Manisha, and Rushed Kanawati. "Link prediction in complex networks by supervised rank aggregation." 2012 IEEE 24th International Conference on Tools with Artificial Intelligence. Vol. 1. IEEE, 2012.</a>

<a id="CRF">[[14]](https://www.jmlr.org/papers/volume15/volkovs14a/volkovs14a.pdf) Volkovs, Maksims N., and Richard S. Zemel. "New learning methods for supervised and unsupervised preference aggregation." The Journal of Machine Learning Research 15.1 (2014): 1135-1176.</a>

<a id="CSRA">[[15]](https://ieeexplore.ieee.org/abstract/document/9053496) Yu, Yinxue, et al. "Crowdsourcing-Based Ranking Aggregation for Person Re-Identification." ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.</a>

<a id="AggRankDe">[[16]](https://www.mdpi.com/2079-9292/11/3/369) Bałchanowski, Michał, and Urszula Boryczka. "Aggregation of Rankings Using Metaheuristics in Recommendation Systems." Electronics 11.3 (2022): 369.</a>

<a id="IRA">[[17]](https://bmvc2022.mpi-inf.mpg.de/0386.pdf) Huang, Ji, et al. "Ranking Aggregation with Interactive Feedback for Collaborative Person Re-identification." (2022).</a>

<a id="QIIRA">[[18]](https://bmvc2022.mpi-inf.mpg.de/0386.pdf) Chunyu Hu, Hong Zhang, Chao Liang, and Huang Hao. QI-IRA: Quantum-inspired interactive ranking aggregation for person re-identification. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 1–9, 2024. (2022).</a>

<a id="semi">[[19]](https://dl.acm.org/doi/abs/10.1145/1458082.1458315) Chen, Shouchun, et al. "Semi-supervised ranking aggregation." Proceedings of the 17th ACM conference on Information and knowledge management. 2008.(2022).</a>

<a id="BDB">[[20]](https://openaccess.thecvf.com/content_ICCV_2019/html/Dai_Batch_DropBlock_Network_for_Person_Re-Identification_and_Beyond_ICCV_2019_paper.html) Dai, Zuozhuo, et al. "Batch dropblock network for person re-identification and beyond." Proceedings of the IEEE/CVF international conference on computer vision. 2019.</a>

<a id="BOT">[[21]](https://openaccess.thecvf.com/content_CVPRW_2019/html/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.html) Luo, Hao, et al. "Bag of tricks and a strong baseline for deep person re-identification." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops. 2019.</a>

<a id="top"> [[22]](https://ieeexplore.ieee.org/abstract/document/9412017) Quispe, Rodolfo, and Helio Pedrini. "Top-db-net: Top dropblock for activation enhancement in person re-identification." 2020 25th International conference on pattern recognition (ICPR). IEEE, 2021.</a>

<a id="light">  [[23]](https://ieeexplore.ieee.org/abstract/document/9506733) Herzog, Fabian, et al. "Lightweight multi-branch network for person re-identification." 2021 IEEE International Conference on Image Processing (ICIP). IEEE, 2021.</a>

<a id="FPB"> [[24]](https://arxiv.org/abs/2108.01901) Zhang, Suofei, et al. "FPB: feature pyramid branch for person re-identification." arXiv preprint arXiv:2108.01901 (2021).</a>

<a id="lu">  [[25]](https://openaccess.thecvf.com/content/CVPR2021/html/Fu_Unsupervised_Pre-Training_for_Person_Re-Identification_CVPR_2021_paper.html) Fu, Dengpan, et al. "Unsupervised pre-training for person re-identification." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.</a>

<a id="market">  [[26]](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html) Zheng, Liang, et al. "Scalable person re-identification: A benchmark." Proceedings of the IEEE international conference on computer vision. 2015.</a>

<a id="duke"> [[27]](https://link.springer.com/chapter/10.1007/978-3-319-48881-3_2) Ristani, Ergys, et al. "Performance measures and a data set for multi-target, multi-camera tracking." European conference on computer vision. Cham: Springer International Publishing, 2016.</a>

<a id="cuhk"> [[28]](https://openaccess.thecvf.com/content_cvpr_2014/html/Li_DeepReID_Deep_Filter_2014_CVPR_paper.html) Li, Wei, et al. "Deepreid: Deep filter pairing neural network for person re-identification." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.</a>

<a id="rank1"> [[29]](https://ieeexplore.ieee.org/abstract/document/9336268) Ye, Mang, et al. "Deep learning for person re-identification: A survey and outlook." IEEE transactions on pattern analysis and machine intelligence 44.6 (2021): 2872-2893.</a>

<a id="RS"> [[30]](https://dl.acm.org/doi/abs/10.1145/3365375) Oliveira, Samuel EL, et al. "Is rank aggregation effective in recommender systems? an experimental analysis." ACM Transactions on Intelligent Systems and Technology (TIST) 11.2 (2020): 1-26.</a>

<a id="lenskit"> [[31]](https://github.com/lenskit/lkpy) Michael D. Ekstrand. 2020. LensKit for Python: Next-Generation Software for Recommender Systems Experiments. In Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM '20). DOI:10.1145/3340531.3412778. arXiv:1809.03125 [cs.IR].</a>

<a id="Kerkentzes">  [[32]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4178426/) Kerkentzes, Konstantinos, et al. "Hidden treasures in “ancient” microarrays: gene-expression portrays biology and potential resistance pathways of major lung cancer subtypes and normal tissue." Frontiers in oncology 4 (2014): 251.</a>

<a id="Li"> [[33]](https://link.springer.com/article/10.1007/s13277-015-3576-y) Li, Yafang, et al. "RNA-seq analysis of lung adenocarcinomas reveals different gene expression profiles between smoking and nonsmoking patients." Tumor Biology 36 (2015): 8993-9003.</a>

<a id="Zhou">  [[34]](https://www.nature.com/articles/onc2016242) Zhou, Y., et al. "microRNAs with AAGUGC seed motif constitute an integral part of an oncogenic signaling network." Oncogene 36.6 (2017): 731-745.</a>
 
<a id="Kim"> [[35]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0055596) Kim, Sang Cheol, et al. "A high-dimensional, deep-sequencing study of lung adenocarcinoma in female never-smokers." PloS one 8.2 (2013): e55596.</a>

<a id="recall"> [[36]](https://academic.oup.com/bioinformatics/article/38/21/4927/6696211) Wang, Bo, et al. "Systematic comparison of ranking aggregation methods for gene lists in experimental results." Bioinformatics 38.21 (2022): 4927-4933.</a>

<a id="IEIR"> [[37]](https://ieeexplore.ieee.org/abstract/document/10391254) Feng, Shiwei, et al. "An Experimental Study of Unsupervised Rank Aggregation Methods in World University Rankings." 2023 International Conference on Intelligent Education and Intelligent Research (IEIR). IEEE, 2023.</a>

 ## Contacts

 If you encounter any problems, you can contact us via email 2021302111226@whu.edu.cn
