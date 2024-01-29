# unsupervised-RA-methods
These unsupervised RA methods were tested on our preprocessed dataset and the results of our preprocessing are in the folder. There are more than 20 of these methods including old RA methods and new RA methods, if there is a need to test other datasets, please follow the comments in the code for dataset preprocessing and code modification.

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


- - -
If you have problems using the code or have suggestions for changes, please contact waii2022@whu.edu.cn
