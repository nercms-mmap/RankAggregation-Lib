# Rank Aggregation (RA) methods
21 unsupervised RA methods, 7 supervised RA methods and 1 semi-supervised RA methods were tested on our preprocessed datasets. These datasets cover the areas of person re-identification (re-ID), recommendation system, bioinformatics and social choices. The tested methods include both classical and state-of-the-art RA methods. If there is a need to test other datasets, please follow the instructions in the code comments for dataset preprocessing and necessary code modifications.

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
                <li>$\textrm{iRANK}$ <a href="#iRANK">[7]</li>
                <li>$\textrm{Mean}$ <a href="#Mean">[8]</li>
                <li>$\textrm{HPA}$ <a href="#HPA&postNDCG">[9]</li>
                <li>$\textrm{PostNDCG}$ <a href="#HPA&postNDCG">[9]</li>
                <li>$\textrm{ER}$ <a href="#ER">[10]</li>
                <li>$\textrm{Mork-H}$ <a href="#Mork-H">[40]</li>
                <li>$\textrm{CG}$ <a href="#CG">[11]</li>
                <li>$\textrm{DIBRA}$ <a href="#DIBRA">[12]</li>
                <li>$\textrm{Borda-Score}$ <a href="#borda_score">[13]</li>
        </td>
        <td>
            <ul>
                <li>$\textrm{wBorda}$ <a href="#wBorda">[14]</li>
                <li>$\textrm{CRF}$ <a href="#CRF">[15]</li>
                <li>$\textrm{CSRA}$ <a href="#CSRA">[16]</li>
                <li>$\textrm{AggRankDE}$ <a href="#AggRankDe">[17]</li>
                <li>$\textrm{IRA}_\textrm{R}$ <a href="#IRA">[18]</li>
                <li>$\textrm{IRA}_\textrm{S}$ <a href="#IRA">[18]</li>
                <li>$\textrm{QI-IRA}$ <a href="#QIIRA">[19]</li>
        </td>
        <td>
            <ul>
                <li>$\textrm{SSRA}$ <a href="#semi">[20]</li>
        </td>
    </tbody>
</table>

# Directory Structure
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
  
![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/43f61aa8-a45f-4501-82e9-13926a85ab5b)

Table 1: Rank@1 (%) and mAP (%) [[29]](#rank1) results for selected feature extraction methods on re-ID datasets.
</div>

we use official training sets to train basic re-ID and fully-supervised RA methods, and use official test sets to evaluate all RA methods.Table 2 presents the parameters of the semi-supervised and supervised methods, along with their type and the value that was set during the re-ID experiments. Note that a parameter setting of default means that for each query in the training set, the value taken is equal to the total number of relevant labels.
<div align="center">
  
![SupPara](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/e65cbb92-416e-454e-af09-9a3eea982c7e)

Table 2: The parameters of the supervised rank aggregation methods.
</div>

Table 3 shows the results of the experiment conducted on the four re-ID datasets, representing the quality of all rank aggregation methods.

<div align="center">

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/138d9ce5-b5d0-414f-ac33-63e2aa423ea5)

Table 3: Rank@1 (%) and mAP (%) [[30]](#RS) results for rank aggregation methods on re-ID datasets.
</div>

### Recommendation System

In recommendation system dataset (MovieLens 1M [[31]](#movielens)), we perform a two-step pre-processing before giving them as input to the recommendation algorithms: (i) Binarization of the ratings of the datasets, as the items are considered only as relevant or irrelevant in the top-N recommendation task, whichs means an item is relevant to a user if its rating is greater than the median of the ratings given by the user. (ii) Removal of users and items that do not reach a predefined threshold value regarding frequency of ratings, which means we removed from the dataset infrequent items and users that rated very few items. Items rated by less than 10 users were removed, together with users that had rated less than 10 items [[30]](#RS).

On MovieLens 1M dataset,we divide the movies evaluated by each user into matrices $M_{train}$ and $M_{test}$. Specifically, we use 60% ratings of each user for $M_{train}$, and the remaining for $M_{test}$. Therefore, two matrices share the same set of users but have different movies recommended for these users. We use two criteria, Rank@1 and mAP@10 criteria, to evaluate the performance.

We select six recommendation algorithms [[32]](#lenskit) (UserUser, BiasedMF, FunkSVD, ImplicitMF, ItemItem, MostPopular) in the experimental phase. All of them are available in the [LensKit](https://github.com/lenskit/lkpy) library. Table 4 presents the parameters of the recommendation algorithms, along with their type and the value that was set during our experiment. The names of the parameters used are consistent with the parameter names available in the LensKit library. We represent the quality of recommendations generated by six recommendation algorithms in Table 5.

<div align="center">

![RecPara](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/606baf2e-ef0a-4a1a-9f62-aa4d45b1a0e3)

Table 4: The parameters of the recommendation algorithms.

![Rec-ini](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/41b3c3ac-d53b-45f4-a74b-0272fd6f46a3)

Table 5: Rank@1 (%) and mAP@10 (%) [[30]](#RS) results for selected recommendation algorithms on MovieLens 1M dataset.
</div>

Different rank aggregation methods will combine the six recommendations into a consensus one. We show their performance in Table 6.

<div align="center">

![image](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/f5c8c706-4a76-4ccd-a7e1-1d6b003b7588)

Table 6: Rank@1 (%) and mAP@10 (%) [[30]](#RS) results for rank aggregation methods on MovieLens 1M datasets
</div>

### Bioinformatics

In bioinformatics, we select a real dataset (NSCLC [[33]](#nsclc)) related to cancer to conduct our experiment. Because there is no labeled data in the NSCLC dataset, we do not measure supervised and semi-supervised RA methods on NSCLC. The NSCLC dataset consists of four basic rankings which are of length 2270, 275, 543, 3501. The sources of them are [[34]](#Kerkentzes), [[35]](#Li), [[36]](#Zhou) and [[37]](#Kim). We consider the recall performance criteria in the aggregated list based on the top 400 and 800 genes. Thus, the result of all unsupervised RA methods is shown in Table 7.

<div align="center">

![NSCLC](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/7dd85d0c-668d-4065-99cb-ac420d7cd8aa)

Table 7: Recall@400 (%) and Recall@800 (%) [[38]](#recall) results for unsupervised RA methods on NSCLC datasets.
</div>

###  Social Choices

We select five popular world university rankings: [[ARWU]](https://www.shanghairanking.com/), [[QS]](https://www.qs.com/rankings-performance/), [[THE]](https://www.int-res.com/abstracts/esep/v13/n2/p125-130/), [[US-NEW]](https://www.usnews.com/best-colleges)  and [[URAP]](https://urapcenter.org/), where there is duplication of top-200 universities. In the five university rankings, because some universities appear in one ranking but not in another, we process the data for these five popular world university rankings. Specifically, we take the rank of the basic university ranked in its basic ranking for the duplicates. Furthermore, We first collect the set of all universities for these five university rankings, and if a university that belongs to the set of all universities does not appear in a particular basic ranking, we set this university to be the 201st of the basic ranking, and so on, until all five university rankings are processed. Eventually, we obtain and aggregate five rankings for an equal number of universities. We measure the normality and the overall impartiality [[39]](#IEIR) to represent the quality of a consensus ranking. The result of all unsupervised RA methods is shown in Table 8.

<div align="center">

![SC](https://github.com/nercms-mmap/RankAggregation-Lib/assets/121333364/cc1d411b-2979-4c4c-94c1-b26faadf5a13)

Table 8: Normality and the overall of impartiality results for unsupervised RA methods on World University Ranking.
</div>

## References
<a id="Comb">[[1]](https://books.google.com.tw/books?hl=zh-CN&lr=&id=W8MZAQAAIAAJ&oi=fnd&pg=PA243&dq=Combination+of+multiple+searches.&ots=3XwVWFAQ5n&sig=EGO4Nkeo5BIsfg0HOpiHsnNPjm4&redir_esc=y#v=onepage&q=Combination%20of%20multiple%20searches.&f=false) Fox, E., & Shaw, J. (1994). Combination of multiple searches. NIST special publication SP, 243-243.</a>

<a id="MC">[[2]](https://dl.acm.org/doi/abs/10.1145/371920.372165) Dwork, C., Kumar, R., Naor, M., & Sivakumar, D. (2001, April). Rank aggregation methods for the web. In Proceedings of the 10th international conference on World Wide Web (pp. 613-622).</a>

<a id="Borda">[[3]](https://dl.acm.org/doi/abs/10.1145/383952.384007) Aslam, J. A., & Montague, M. (2001, September). Models for metasearch. In Proceedings of the 24th annual international ACM SIGIR conference on Research and development in information retrieval (pp. 276-284).</a>

<a id="Dowdall">[[4]](https://journals.sagepub.com/doi/abs/10.1177/0192512102023004002) Reilly, B. (2002). Social choice in the south seas: Electoral innovation and the borda count in the pacific island countries. International Political Science Review, 23(4), 355-372.</a>

<a id="Median">[[5]](https://dl.acm.org/doi/abs/10.1145/872757.872795) Fagin, R., Kumar, R., & Sivakumar, D. (2003, June). Efficient similarity search and classification via rank aggregation. In Proceedings of the 2003 ACM SIGMOD international conference on Management of data (pp. 301-312).</a>

<a id="RRF">[[6]](https://dl.acm.org/doi/abs/10.1145/1571941.1572114) Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009, July). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval (pp. 758-759).</a>

<a id="iRANK">[[7]](https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/asi.21296) Wei, F., Li, W., & Liu, S. (2010). iRANK: A rank‐learn‐combine framework for unsupervised ensemble ranking. Journal of the American Society for Information Science and Technology, 61(6), 1232-1243.</a>

<a id="Mean">[[8]](https://proceedings.mlr.press/v14/burges11a/burges11a.pdf) Burges, C., Svore, K., Bennett, P., Pastusiak, A., & Wu, Q. (2011, January). Learning to rank using an ensemble of lambda-gradient models. In Proceedings of the learning to rank Challenge (pp. 25-35). PMLR.</a>

<a id="HPA&postNDCG">[[9]](https://link.springer.com/chapter/10.1007/978-3-030-45442-5_17) Fujita, S., Kobayashi, H., & Okumura, M. (2020). Unsupervised Ensemble of Ranking Models for News Comments Using Pseudo Answers. In Advances in Information Retrieval: 42nd European Conference on IR Research, ECIR 2020, Lisbon, Portugal, April 14–17, 2020, Proceedings, Part II 42 (pp. 133-140). Springer International Publishing.</a>

<a id="ER">[[10]](https://www.sciencedirect.com/science/article/pii/S0305048319308448) Mohammadi, M., & Rezaei, J. (2020). Ensemble ranking: Aggregation of rankings produced by different multi-criteria decision-making methods. Omega, 96, 102254.</a>

<a id="CG">[[11]](https://www.tandfonline.com/doi/abs/10.1080/01605682.2019.1657365) Xiao, Y., Deng, H. Z., Lu, X., & Wu, J. (2021). Graph-based rank aggregation method for high-dimensional and partial rankings. Journal of the Operational Research Society, 72(1), 227-236.</a>

<a id="DIBRA">[[12]](https://www.sciencedirect.com/science/article/abs/pii/S0957417422007710) Akritidis, L., Fevgas, A., Bozanis, P., & Manolopoulos, Y. (2022). An unsupervised distance-based model for weighted rank aggregation with list pruning. Expert Systems with Applications, 202, 117435.</a>

<a id="wBorda">[[13]](https://ieeexplore.ieee.org/abstract/document/6495123) Pujari, M., & Kanawati, R. (2012, November). Link prediction in complex networks by supervised rank aggregation. In 2012 IEEE 24th International Conference on Tools with Artificial Intelligence (Vol. 1, pp. 782-789). IEEE.</a>

<a id="CRF">[[14]](https://www.jmlr.org/papers/volume15/volkovs14a/volkovs14a.pdf) Volkovs, M. N., & Zemel, R. S. (2014). New learning methods for supervised and unsupervised preference aggregation. The Journal of Machine Learning Research, 15(1), 1135-1176.</a>

<a id="CSRA">[[15]](https://ieeexplore.ieee.org/abstract/document/9053496) Yu, Y., Liang, C., Ruan, W., & Jiang, L. (2020, May). Crowdsourcing-Based Ranking Aggregation for Person Re-Identification. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1933-1937). IEEE.</a>

<a id="AggRankDe">[[16]](https://www.mdpi.com/2079-9292/11/3/369) Bałchanowski, M., & Boryczka, U. (2022). Aggregation of Rankings Using Metaheuristics in Recommendation Systems. Electronics, 11(3), 369.</a>

<a id="IRA">[[17]](https://bmvc2022.mpi-inf.mpg.de/0386.pdf) Huang, J., Liang, C., Zhang, Y., Wang, Z., & Zhang, C. (2022). Ranking Aggregation with Interactive Feedback for Collaborative Person Re-identification.</a>

<a id="QIIRA">[[18]](https://aaai.org/wp-content/uploads/2024/01/AAAI_Main-Track_2024-01-04.pdf) Hu, C., Zhang, H., Liang, C., & Huang, H. (2024). QI-IRA: Quantum-inspired interactive ranking aggregation for person re-identification. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 38, pp. 1-9).</a>

<a id="semi">[[19]](https://dl.acm.org/doi/abs/10.1145/1458082.1458315) Chen, S., Wang, F., Song, Y., & Zhang, C. (2008, October). Semi-supervised ranking aggregation. In Proceedings of the 17th ACM conference on Information and knowledge management (pp. 1427-1428).</a>

<a id="BDB">[[20]](https://openaccess.thecvf.com/content_ICCV_2019/html/Dai_Batch_DropBlock_Network_for_Person_Re-Identification_and_Beyond_ICCV_2019_paper.html) Dai, Z., Chen, M., Gu, X., Zhu, S., & Tan, P. (2019). Batch dropblock network for person re-identification and beyond. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 3691-3701).</a>

<a id="BOT">[[21]](https://openaccess.thecvf.com/content_CVPRW_2019/html/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.html) Luo, H., Gu, Y., Liao, X., Lai, S., & Jiang, W. (2019). Bag of tricks and a strong baseline for deep person re-identification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops (pp. 0-0).</a>

<a id="top"> [[22]](https://ieeexplore.ieee.org/abstract/document/9412017) Quispe, R., & Pedrini, H. (2021, January). Top-db-net: Top dropblock for activation enhancement in person re-identification. In 2020 25th International conference on pattern recognition (ICPR) (pp. 2980-2987). IEEE.</a>

<a id="light">  [[23]](https://ieeexplore.ieee.org/abstract/document/9506733) Herzog, F., Ji, X., Teepe, T., Hörmann, S., Gilg, J., & Rigoll, G. (2021, September). Lightweight multi-branch network for person re-identification. In 2021 IEEE International Conference on Image Processing (ICIP) (pp. 1129-1133). IEEE.</a>

<a id="FPB"> [[24]](https://arxiv.org/abs/2108.01901) Zhang, S., Yin, Z., Wu, X., Wang, K., Zhou, Q., & Kang, B. (2021). FPB: feature pyramid branch for person re-identification. arXiv preprint arXiv:2108.01901.</a>

<a id="lu">  [[25]](https://openaccess.thecvf.com/content/CVPR2021/html/Fu_Unsupervised_Pre-Training_for_Person_Re-Identification_CVPR_2021_paper.html) Fu, D., Chen, D., Bao, J., Yang, H., Yuan, L., Zhang, L., ... & Chen, D. (2021). Unsupervised pre-training for person re-identification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 14750-14759).</a>

<a id="market">  [[26]](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html) Zheng, L., Shen, L., Tian, L., Wang, S., Wang, J., & Tian, Q. (2015). Scalable person re-identification: A benchmark. In Proceedings of the IEEE international conference on computer vision (pp. 1116-1124).</a>

<a id="duke"> [[27]](https://link.springer.com/chapter/10.1007/978-3-319-48881-3_2) Ristani, E., Solera, F., Zou, R., Cucchiara, R., & Tomasi, C. (2016, October). Performance measures and a data set for multi-target, multi-camera tracking. In European conference on computer vision (pp. 17-35). Cham: Springer International Publishing.</a>

<a id="cuhk"> [[28]](https://openaccess.thecvf.com/content_cvpr_2014/html/Li_DeepReID_Deep_Filter_2014_CVPR_paper.html) Li, W., Zhao, R., Xiao, T., & Wang, X. (2014). Deepreid: Deep filter pairing neural network for person re-identification. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 152-159).</a>

<a id="rank1"> [[29]](https://ieeexplore.ieee.org/abstract/document/9336268) Ye, M., Shen, J., Lin, G., Xiang, T., Shao, L., & Hoi, S. C. (2021). Deep learning for person re-identification: A survey and outlook. IEEE transactions on pattern analysis and machine intelligence, 44(6), 2872-2893.</a>

<a id="RS"> [[30]](https://dl.acm.org/doi/abs/10.1145/3365375) Oliveira, S. E., Diniz, V., Lacerda, A., Merschmanm, L., & Pappa, G. L. (2020). Is rank aggregation effective in recommender systems? an experimental analysis. ACM Transactions on Intelligent Systems and Technology (TIST), 11(2), 1-26.</a>

<a id="movielens"> [[31]](https://dl.acm.org/doi/abs/10.1145/2827872) Harper, F. M., & Konstan, J. A. (2015). The movielens datasets: History and context. Acm transactions on interactive intelligent systems (tiis), 5(4), 1-19.</a>

<a id="lenskit"> [[32]](https://github.com/lenskit/lkpy) Ekstrand, M. D. (2020, October). Lenskit for python: Next-generation software for recommender systems experiments. In Proceedings of the 29th ACM international conference on information & knowledge management (pp. 2999-3006).</a>

<a id="nsclc"> [[33]](https://academic.oup.com/bioinformatics/article/38/21/4927/6696211?login=false) Wang, B., Law, A., Regan, T., Parkinson, N., Cole, J., Russell, C. D., ... & Baillie, J. K. (2022). Systematic comparison of ranking aggregation methods for gene lists in experimental results. Bioinformatics, 38(21), 4927-4933.</a>

<a id="Kerkentzes">  [[34]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4178426/) Kerkentzes, K., Lagani, V., Tsamardinos, I., Vyberg, M., & Røe, O. D. (2014). Hidden treasures in “ancient” microarrays: gene-expression portrays biology and potential resistance pathways of major lung cancer subtypes and normal tissue. Frontiers in oncology, 4, 251.</a>

<a id="Li"> [[35]](https://link.springer.com/article/10.1007/s13277-015-3576-y) Li, Y., Xiao, X., Ji, X., Liu, B., & Amos, C. I. (2015). RNA-seq analysis of lung adenocarcinomas reveals different gene expression profiles between smoking and nonsmoking patients. Tumor Biology, 36, 8993-9003.</a>

<a id="Zhou">  [[36]](https://www.nature.com/articles/onc2016242) Zhou, Y., Frings, O., Branca, R. M., Boekel, J., le Sage, C., Fredlund, E., ... & Orre, L. M. (2017). microRNAs with AAGUGC seed motif constitute an integral part of an oncogenic signaling network. Oncogene, 36(6), 731-745.</a>
 
<a id="Kim"> [[37]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0055596) Kim, S. C., Jung, Y., Park, J., Cho, S., Seo, C., Kim, J., ... & Lee, S. (2013). A high-dimensional, deep-sequencing study of lung adenocarcinoma in female never-smokers. PloS one, 8(2), e55596.</a>

<a id="recall"> [[38]](https://academic.oup.com/bioinformatics/article/38/21/4927/6696211) Wang, B., Law, A., Regan, T., Parkinson, N., Cole, J., Russell, C. D., ... & Baillie, J. K. (2022). Systematic comparison of ranking aggregation methods for gene lists in experimental results. Bioinformatics, 38(21), 4927-4933.</a>

<a id="IEIR"> [[39]](https://ieeexplore.ieee.org/abstract/document/10391254) Feng, S., Deng, Q., Wang, S., Song, L., & Liang, C. (2023, November). An Experimental Study of Unsupervised Rank Aggregation Methods in World University Rankings. In 2023 International Conference on Intelligent Education and Intelligent Research (IEIR) (pp. 1-8). IEEE.</a>

<a id="Mork-H">[[40]](https://www.sciencedirect.com/science/article/abs/pii/S0377221719307039) Ivano Azzini., & Giuseppe Munda. (2020). Azzini, I., & Munda, G. (2020). A new approach for identifying the Kemeny median ranking. European Journal of Operational Research, 281(2), 388-401. </a>

 ## Contacts

 If you encounter any problems, you can contact us via email 2021302111226@whu.edu.cn
