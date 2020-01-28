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

# UNSW-NB15 Dataset Feature Selection and Network Intrusion Detection using Deep Learning
V. Kanimozhi, Prem Jacob. Jan 2019

Chose only four best features sbytes, sttl, sload, ct_dst_src_ltm using RFE algorithm.

![roc curve](roc%20curve%20with%20neural%20network.png)
![ann report](Classification%20report%20of%20ANN.png)
![confusion matrix](confusion%20matrix%20%20of%20ANN.png)

Limitations:
* No mention of hypertuning
* Only feature selection , no feature preprocessing was done.
* Mentioned MLP in ARTIFICIAL NEURAL NETWORKS section, but no mention of actually 
which neural network was used and its structure.

# Using machine learning techniques to identify rare cyber‐attacks on the UNSW‐NB15 dataset
Sikha Bagui1 Ezhil Kalaimannan1 Subhash Bagui2 Debarghya Nandi3 Anthony Pinto. Oct 2019

# An Ensemble Intrusion Detection Technique based on proposed Statistical Flow Features for Protecting Network Traffic of Internet of Things
Nour Moustafa. 2018

The overall performance evaluation of the DT, NB, ANN
and the suggested ensemble method in terms of Accuracy
(Acc), DR, FPR and processing time (Time) is discussed
using the DNS and HTTP data sources of the UNSW-NB15
and NIMS botnet datasets. 

So they combined both dataset and separately gave predictions for DNS data and HTTP data.

# Enhanced Network Anomaly Detection Based on Deep Neural Networks
SHERAZ NASEER1,2, YASIR SALEEM1 ... September 21, 2018

![comparison](comparison%20of%20NSLKDD%20dataset%20results.png)

# NIDS using Machine Learning Classifiers on UNSW-NB15 and KDDCUP99 Datasets
Dipali Gangadhar Mogal1, Sheshnarayan R. Ghungrad2, Bapusaheb B. Bhusare3. 2017

No mention of whether the result is on train data or test data. Also which validation process was followed
isn't mentioned anywhere in the paper.

![NB and LR](nb%20and%20lr%20for%20unsw.png)

# Towards Developing Network forensic mechanism for Botnet Activities in the IoT based on Machine Learning Techniques
Nickilaos Koroniotis1, Nour Moustafa1, Elena Sitnikova1, Jill Slay1. 2017

The portion of it that we will be making use, contains 257,673 (created by combining the training and
testing datasets). The Weka tool was used for applying the four techniques using the default parameters with a 10-fold cross
validation in order to effectively measure their performance.

To test the classifiers, we performed Information Gain Ranking Filter (IG) for
selecting the highest ten ranked features as listed in Table 2.

![top features](top%20ten%20features.png)


Decision Tree C4.5 Classifier was the best at distinguishing between Botnet and normal network traffic.

![four techniques](performance%20evaluation%20of%20four%20techniques.png)

Limitations:
* Why mix train and test ?
* Why 10 features , why not 9/11?
* The ANN specified was just a perceptron model. So no actual deep learning models were used.
* No mention of hypertuning process.
* Only binary classification


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


# Classification Performance Improvement of UNSW-NB15 Dataset Based on Feature Selection
2019

Couldn't understand anything, as it is written in korean.

# The significant features of the UNSW-NB15 and the KDD99 data sets for Network Intrusion Detection Systems
By Nour Moustofa. Nov 2015

Very poor results

![evaluation](evaluataion%20of%20unsw-nb15%20with%20best%20features.png)

# The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set
By Nour Moustofa. Jan 2016

Not sure whether the results are on train data or test data.

![comparison](comparison%20between%20KDD99%20and%20UNSW-NB15.png)


# Feature Selection in UNSW-NB15 and KDDCUP’99 datasets
Tharmini Janarthanan, Shahrzad Zargari . 2017