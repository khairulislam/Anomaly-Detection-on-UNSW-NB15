# notebook https://www.kaggle.com/khairulislam/ten-fold-cross-validation-with-different-models
import warnings
warnings.simplefilter(action='ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

root = '../input/unsw-nb15/'
train = pd.read_csv(root + 'UNSW_NB15_training-set.csv')


def get_cat_columns(df):
    return [col for col in df.columns if df[col].dtype == 'object']


def feature_process(df):
    df.loc[~df['state'].isin(['FIN', 'INT', 'CON', 'REQ', 'RST']), 'state'] = 'others'
    df.loc[~df['service'].isin(['-', 'dns', 'http', 'smtp', 'ftp-data', 'ftp', 'ssh', 'pop3']), 'service'] = 'others'
    df.loc[df['proto'].isin(['igmp', 'icmp', 'rtp']), 'proto'] = 'igmp_icmp_rtp'
    df.loc[~df['proto'].isin(['tcp', 'udp', 'arp', 'ospf', 'igmp_icmp_rtp']), 'proto'] = 'others'
    return df


# Pre-processing . Remove this whole block if the data is already preprocessed
drop_columns = ['attack_cat', 'id']
train.drop(drop_columns, axis=1, inplace=True)
# separate features and labels
x, y = train.drop(['label'], axis=1), train['label']
# feature pre-process
x = feature_process(x)
categorical_columns = get_cat_columns(x)
non_categorical_columns = [col for col in x.columns if col not in categorical_columns]
# scaling
scaler = StandardScaler()
x[non_categorical_columns] = scaler.fit_transform(x[non_categorical_columns])
# one hot encoding
x = pd.get_dummies(x)

# ten fold cross validation
folds = 10
seed = 1
kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
models = [
    RandomForestClassifier(random_state=seed),
    LogisticRegression(random_state=seed),
    GradientBoostingClassifier(random_state=seed),
    DecisionTreeClassifier(random_state=seed)
]

results = {
    "model": [],
    "accuracy": [],
    "f1": []
}

for model in models:
    results["model"].append(model.__class__.__name__)
    scores = cross_validate(model, x, y, cv=kf, scoring=["accuracy", "f1"])
    for key in scores.keys():
        mean, std = scores[key].mean(), scores[key].std()
        print(f"{key}, mean {mean}, std {std}")
    results["accuracy"].append(scores["test_accuracy"].mean())
    results["f1"].append(scores["test_f1"].mean())
    print()

# save validation results
pd.DataFrame(results).to_csv("results.csv", index=False)