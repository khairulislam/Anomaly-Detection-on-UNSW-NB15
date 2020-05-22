# EDA 
https://www.kaggle.com/khairulislam/unsw-nb15-eda

# Data preprocessing
https://www.kaggle.com/khairulislam/data-preprocessing

# Feature Importance
https://www.kaggle.com/khairulislam/unsw-nb15-feature-importance
Filename: importance.csv

# Model selection
Using ten-fold cross validation on popular machine learning models 
to find the best one.
https://www.kaggle.com/khairulislam/ten-fold-cross-validation-with-different-models

| Model | Accuracy | F1 |
 --- | ---  | ----
LogisticRegression	| 0.9354286717347297	| 0.9542239342896803
GradientBoostingClassifier	| 0.9458426691403649	| 0.9611062449808958
DecisionTreeClassifier	| 0.9498805218353074	| 0.9631526109252336
RandomForestClassifier	| 0.9607678800687012	| 0.9713978736211478
LighGBM | 0.961811555768474 | 0.9721410918894631


# Hyper-tuning

# Experiments
Links :
* https://www.kaggle.com/khairulislam/unsw-nb15-lightgbm
* https://www.kaggle.com/khairulislam/unsw-nb15-witih-randomforest
## Train performance
Performance of the model on the same dataset on which it was trained.



## Ten-fold cross validation
Using Stratified kfold cross validation to validate model performance.
### Train
### Test

### Combined (train + test)

## Train test validation
Training the model on train dataset. Then testing on separate test dataset. 

* [Deep Learning Approach for Intelligent Intrusion Detection System](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8681044) :
DNN (4 layers) acc 0.765, pre 0.946, rec 0.695, f1 .801 . RF acc .903, pre .988, rec 0.867, f1 0.924.
* [Feasibility of Supervised Machine Learning for Cloud
Security](https://arxiv.org/ftp/arxiv/papers/1810/1810.09878.pdf):
Logistic Regression acc 89.26%, TP 93.7% TN 95.7% at prediction threshold 0.5. Increasing
prediction threshold to 0.7-0.8 TP 97%, but TN 80%.

* [A Two-Stage Classifier Approach using RepTree Algorithm for Network Intrusion Detection](https://www.researchgate.net/profile/Mustapha_Belouch2/publication/318099406_A_Two-Stage_Classifier_Approach_using_RepTree_Algorithm_for_Network_Intrusion_Detection/links/5b2e227c4585150d23c66a27/A-Two-Stage-Classifier-Approach-using-RepTree-Algorithm-for-Network-Intrusion-Detection.pdf):
Used a decision tree named REPTree (Reduced Error Pruning Tree) to get accuracy 88.95%.
* [Network Intrusion Detection in Big Dataset Using Spark](file:///E:/Git%20projects/Anomaly%20Detection%20on%20UNSW-NB15/Related%20papers/Network%20Intrusion%20Detection%20in%20Big%20Dataset%20Using%20Spark.pdf):
Using Dimension Reduction (LDA) on the dataset using spark then the performance of REP Tree
acc 93.56%, FPR 2.3%, prec 83.3%, rec 83.2%, roc 90.5%.

# Papers that weren't compared
* [Deep Learning Approach for Cyberattack Detection](https://www.researchgate.net/profile/Liyuan_Liu23/publication/326563074_Deep_learning_approach_for_cyberattack_detection/links/5c01abf092851c63cab2aabb/Deep-learning-approach-for-cyberattack-detection.pdf):
The two datasets are randomly splitted using the same rule.
80% of data was used to fit DFEL and get the pre-trained
model. The remaining 20% of the data was randomly split
into 70%/30% as training/testing data for classifiers.

* Building an Effective Intrusion Detection SystemUsing the Modified Density Peak ClusteringAlgorithm and Deep Belief Networks:
Multiclass classification

* [A New Generalized Deep Learning Framework
Combining Sparse Autoencoder and Taguchi Method for
Novel Data Classification and Processing](http://downloads.hindawi.com/journals/mpe/2018/3145947.pdf): only worked with DDoS datasete.
* [An Empirical Evaluation of Deep Learningfor Network Anomaly Detection](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8846674):
Mentioned 100% result of all metrics (acc, pre, rec, f1) for NSL-KDD, KYOTO-HONEYPOT, UNSW-NB15, IDS2017. Used seq2sep 
model. For unsw-nb15 used  train as test and test as train.

* [Intrusion Detection Using Big Data and Deep Learning Techniques](https://dl.acm.org/doi/pdf/10.1145/3299815.3314439):
Used the big dataset of UNSW-NB15 with five fold cross validation.
* [An Effective Deep Learning Based Scheme forNetwork Intrusion Detection](https://ieeexplore.ieee.org/abstract/document/8546162):
In this dataset, there are 2.54 million samples in total,
containing 9 types of attack samples and 2.2 million normal
samples. Each sample has 47 features. We randomly assign
them into two sets for training and testing, respectively, each
of which contains 1.905 million and 0.635 million samples.
The ratios of normal vs. attack samples of both sets are 6.9,
remaining the same as in the original dataset.