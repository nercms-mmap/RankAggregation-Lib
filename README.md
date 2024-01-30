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

## Result for datasets

### Information retrieval

### Bioinformatics
| Method          | Recall@200 | Recall@400 | Recall@600 | Recall@800 | Recall@1000 | NDCG@500 |
|:----------------------:|:------------:|:------------:|:------------:|:------------:|:-------------:|:----------:|
| CG              | 0.029      | 0.058      | 0.116      | 0.174      | 0.203       | 0.052    |
| CombANZ         | 0.072      | 0.087      | 0.145      | 0.174      | 0.217       | 0.065    |
| CombMAX         | 0.029      | 0.043      | 0.058      | 0.116      | 0.159       | 0.049    |
| CombMIN         | 0          | 0.014      | 0.029      | 0.072      | 0.087       | 0.008    |
| CombMNZ         | 0.072      | 0.087      | 0.145      | 0.174      | 0.217       | 0.065    |
| CombSUM         | 0.072      | 0.087      | 0.145      | 0.174      | 0.217       | 0.065    |
| PostNDCG        | 0.101      | 0.145      | 0.159      | 0.159      | 0.159       | 0.089    |
| BordaCount      | 0.029      | 0.058      | 0.116      | 0.174      | 0.203       | 0.052    |
| Dowdall_method  | 0.058      | 0.145      | 0.203      | 0.275      | 0.29        | 0.113    |
| Mean            | 0.159      | 0.217      | 0.304      | 0.377      | 0.406       | 0.148    |
| Medium          | 0.13       | 0.159      | 0.188      | 0.232      | 0.275       | 0.113    |
| RRF             | 0.043      | 0.116      | 0.13       | 0.261      | 0.275       | 0.094    |
| ER              | 0.029      | 0.058      | 0.116      | 0.174      | 0.203       | 0.052    |
| hparunme        | 0.058      | 0.116      | 0.188      | 0.203      | 0.217       | 0.094    |
| iRANK           | 0.159      | 0.217      | 0.304      | 0.377      | 0.406       | 0.148    |
| MC1             | 0.029      | 0.043      | 0.087      | 0.13       | 0.203       | 0.069    |
| MC2             | 0.029      | 0.058      | 0.101      | 0.174      | 0.203       | 0.078    |
| MC3             | 0.029      | 0.058      | 0.072      | 0.087      | 0.101       | 0.032    |
| MC4             | 0.014      | 0.029      | 0.058      | 0.087      | 0.087       | 0.048    |
| Condorcet       | 0.159      | 0.217      | 0.348      | 0.377      | 0.391       | 0.171    |
| Outranking Approach | 0.159 | 0.261      | 0.319      | 0.333      | 0.333       | 0.179    |
| RRA             | 0.116      | 0.203      | 0.275      | 0.319      | 0.348       | 0.171    |
| WT-INDEG        | 0.058      | 0.087      |0.116       | 0.188      |0.203        |0.058     |
| Agglomerative   | 0.029      | 0.072      | 0.13       | 0.174      | 0.203       | 0.058    |

### Social choice

| Method    | Ice-Flavors Time(ms) | Ice-Flavors KSD | CWS-15-06(SOC) Time(ms) | CWS-15-06(SOC) KSD | ED-11-01(SOC) Time(ms) | ED-11-01(SOC) KSD |
|:-----------:|:----------------------:|:-----------------:|:-------------------------:|:--------------------:|:------------------------:|:-------------------:|
| CG        | 8.37                 | 8               | 8.15                    | 10716              | 8.83                   | 37422             |
| CombANZ   | 9.08                 | 8               | 11.2                    | 17713              | 10.1                   | 114027            |
| CombMAX   | 9.21                 | 10              | 9.33                    | 14420              | 9.3                    | 47392             |
| CombMIN   | 9.03                 | 10              | 8.08                    | 15090              | 9.31                   | 60888             |
| CombMNZ   | 9.78                 | 8               | 9.86                    | 16942              | 11.1                   | 113667            |
| CombSUM   | 10                   | 8               | 9.66                    | 10730              | 10.4                   | 37468             |
| PostNDCG  | 9.39                 | 8               | 9.9                     | 10568              | 9.49                   | 33302             |
| BordaCount| 8.3                  | 8               | 7.6                     | 10730              | 9.3                    | 37468             |
| Dowdall_method    | 9.19     | 8       | 9.06     | 12362   | 9.32     | 37382  |
| Mean              | 8.82     | 8       | 8.9      | 10730   | 9.33     | 37468  |
| Medium            | 9.19     | 8       | 8.7      | 10047   | 9.83     | 32232  |
| RRF               | 9.41     | 8       | 9.72     | 10964   | 9.88     | 36370  |
| ER                | 9.08     | 8       | 9.04     | 10714   | 9.86     | 37428  |
| hparunme          | 8.57     | 10      | 11.7     | 10690   | 9.64     | 38422  |
| iRANK             | 8.56     | 8       | 9.62     | 10722   | 9.82     | 37438  |
| Condorcet         | 6.05     | 8       | 7.19     | 9888    | 13.9     | 31826  |
| Copeland          | 5.18     | 8       | 6.45     | 9888    | 7.15     | 31826  |
| Outranking Approach | 5.35   | 8       | 6.24     | 10662   | 8.13     | 37152  |
| MC1               | 7.35     | 10      | 8.33     | 14092   | 11.5     | 50266  |
| MC2               | 5.56     | 8       | 6.08     | 9906    | 9.38     | 31864  |
| MC3               | 4.77     | 26      | 6.76     | 34238   | 9.34     | 202254 |
| MC4               | 4.86     | 8       | 7.4      | 9906    | 10.7     | 31864  |
| MCT               | 4.7      | 10      | 6.95     | 18324   | 10.1     | 85716  |
| RRA-Exact         | 4.95     | 8       | 6.75     | 11810   | 8.63     | 41054  |
| RRA               | 4.63     | 12      | 6.19     | 13062   | 7.3      | 40356  |
| PrefRel           | 4.64     | 8       | 7.38     | 10722   | 21.4     | 37468  |
| Agglomerative     | 10.1     | 10      | 14       | 13174   | 20.7     | 49640  |
| DIBRA             | 4.7      | 10      | 8.6      | 14076   | 16.5     | 61874  |
| DIBRA-Prune       | 5.43     | 6       | 6.11     | 14996   | 7.16     | 39448  |

### Recommender system


## Reference
[1] 

[2] 

[3] 
- - -
If you have problems using the code or have suggestions for changes, please contact waii2022@whu.edu.cn
