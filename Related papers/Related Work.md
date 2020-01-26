# Anomaly Detection System using Beta Mixture Models and Outlier Detection
By Nour Moustafa.December 2017

Proposes a Beta Mixture Model based Anomaly Detection System on UNSW-NB15 dataset.
The proposed technique was assessed using the eight features from the UNSW-NB15 dataset. 
The features are ct_dst_sport_ltm, tcprtt, dwin, ct_src_dport_ltm,
ct_dst_src_ltm, ct_dst_ltm, smean, service.

To conduct the experiments the dataset, they selected random samples from the UNSW-NB15
dataset with various sample sizes between 50,000 and 200,000. In each sample,
normal instances were approximately 55-65% of the total size, with some used
to create the normal prole and the testing set. Best result was achieved with w value 3 with accuracy 93.4%, 
detection rate 92.7% and false positive rate 5.9%. 

However, as their train test are random, it is impossible to reproduce their results.

# Network Based Intrusion Detection Using the UNSW-NB15 Dataset
Souhail Meftah1, Tajjeeddine Rachidi1 and Nasser Assem. Sep 2019.

Their study consists of both binary and multiclass classification on UNSW-NB15 train and test dataset.
* Preprocessing : Dropped service column for having many missing values
* Feature selection: Using 10-fold cross validation on RandomForest classifier. They found five best features :
ct_dst_src_ltm, ct_scv_dst, ct_dst_sport_ltm, ct_src_dport_ltm, ct_srv_src. 
### Binary classification
They used 10-fold cross validation in train dataset but didn't predict on test dataset. As some columns had
new labels on test data. They argued this is a notable limitation of the training / testing set distribution of this dataset (They could just not consider those columns in those cases).
    * Logistic Regression: AUC = 0.884385, Accuracy 77.21% . (Where are other metrics ?) 
    * Gradient Boosting Machine: logloss 0.15671. Accuracy 61.83% (Again no mention of other metrics)
    * SVM: Accuracy 82.11% .
### Multiclass classification
As SVM was found to be performing best during binary classification, it was picked to feed its results
to a multiclass classifier in stage two. The figures show that C5.0 performed the best.
![Accuracy comparison](accuracy%20comparison.png)
![multiclass f1-score](multiclass%20classifiers%20f1-score.png)
