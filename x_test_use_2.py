import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rc('font', size=14)
import numpy as np
from sklearn.decomposition import PCA

#from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)

#
#working_dir = '/Users/ljyi/Desktop/capstone/capstone8/RF'
working_dir = '/Users/chengpingchai/GoogleDrive/Documents/LJY/UVA/2018_Fall/capstone'
os.chdir(working_dir)
#
raw_data = pd.read_csv('moss_plos_one_data.csv')
raw_data.columns = raw_data.columns.str.replace('.', '_')
raw_data.shape
# (2217958, 62)
col_names = raw_data.columns.tolist()

#==============================================================================
#                             Data Preprocessing
#==============================================================================
# find missing values
df = raw_data
df.head()
df.columns
df_nan = df.isnull().sum(axis=0).to_frame()
df_nan.columns=['counts']
col_nan = df_nan[df_nan['counts']>0]
col_nan_index = list(col_nan.index)

# find unique values in 'id'
id_unique = df['id'].unique().tolist()
id_unique
len(id_unique)
# 8105

# get train and test index based on unique 'id'
import random
random.seed(1)
train_id = random.sample(id_unique, 5674)
test_id = [avar for avar in id_unique if avar not in train_id]

# get rid of variables with two many missing values
data_df = raw_data
drop_cols = ['n_evts', 'LOS', 'ICU_Pt_Days', 'Mort', 'age', 'race', 'svc']  # why not age?
data_df.drop(col_nan_index, inplace=True, axis=1)
data_df.drop(drop_cols, inplace=True, axis=1)

## 'race' with three levels and 'svc' with four levels are categorical data
#dummy_race = pd.get_dummies(data_df['race'])
#data_df_dummy = pd.concat([data_df, dummy_race], axis=1)
#data_df_dummy.drop(columns=['race', 'oth'], inplace=True, axis=1) # dummy variable trap
#
#dummy_svc = pd.get_dummies(data_df['svc'])
#df_svc_dummy = pd.concat([data_df_dummy, dummy_svc], axis=1)
#df_svc_dummy.drop(columns=['svc', 'Other'], inplace=True, axis=1)

list(data_df.columns)
df_dummy = data_df

# split data into training and testing sets
df_dummy.set_index('id', inplace=True)
X_y_train = df_dummy.loc[train_id]
X_y_test = df_dummy.loc[test_id]

# sample training set
true_index = np.where(X_y_train['y'].values.flatten() == True)[0]
false_index = np.where(X_y_train['y'].values.flatten() == False)[0]
random.seed(0)
selected_false_index = random.sample(list(false_index), len(true_index)*2)
train_index = list(np.append(true_index, selected_false_index))
#
#true_index = np.where(X_y_test['y'].values.flatten() == True)[0]
#false_index = np.where(X_y_test['y'].values.flatten() == False)[0]
#random.seed(0)
#selected_false_index = random.sample(list(false_index), len(true_index)*2)
#test_index = list(np.append(true_index, selected_false_index))
# 
X_train = X_y_train.iloc[train_index, X_y_train.columns != 'y']
y_train = X_y_train.iloc[train_index, X_y_train.columns == 'y']
X_test = X_y_test.iloc[:, X_y_test.columns != 'y']
y_test = X_y_test.iloc[:, X_y_test.columns == 'y']
y_test = y_test.values.flatten()

len(y_train)
#1520840
np.sum(y_train == True)
# 16391
np.sum(y_train == False)
# 1504449
np.sum(y_test == True)
# 7490
np.sum(y_test == False)
# 689628

train_col_names = X_train.columns

# over-sampling using SMOTE-Synthetic Minority Oversampling Technique
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
os_data_X, os_data_y = os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X, columns=train_col_names)
os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])

# check the lengths of data now
os_data_X.shape
# (2996702, 55)
len(os_data_y)
# 2996702
# percent of True
n_total = len(os_data_y)
n_true = sum(os_data_y['y']==True)
n_true
# 1498351 (before oversampling: 23881)

n_false = sum(os_data_y['y']==False)
n_false
# 1498351 (before oversampling:2194077)

pct_true = n_true/n_total
pct_true
# 0.5
# 50% are event
pct_false = n_false/n_total
pct_false
# 0.5
# 50% are non-event
# here, the ratio of event to non-event is 1:1 after SMOTE.

# Final data for training 
X_train_balanced = os_data_X
y_train_balanced = os_data_y

n_rows_total = len(y_train_balanced)
#n_rows_total_ls = range(n_rows_total)
random.seed(1)
#sample_rows_index = random.sample(n_rows_total_ls, 100000)
X_train_df = X_train_balanced
y_train_sample = y_train_balanced
y_train_sample = y_train_sample.values.flatten()

# add shock index
SI_train = X_train_df['Pulse']/X_train_df['SBP']
SI_test = X_test['Pulse']/X_test['SBP']
X_train_df['SI'] = SI_train
X_test['SI'] = SI_test

## get column names of X_test
cols_test = X_test.columns

X_test_raw = X_test
# feature scaling
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
sc = StandardScaler()
X_train_sample = sc.fit_transform(X_train_df)  
X_test = sc.transform(X_test)
type(X_train_sample)

# change array to dataframe adding column names
X_train_sample = pd.DataFrame(X_train_sample)
X_train_sample.columns = X_train_df.columns
#y_train_sample = pd.DataFrame(y_train_sample)
#y_train_sample.columns = ['y']
X_test = pd.DataFrame(X_test)
X_test.columns = cols_test
#
forest_clf = RandomForestClassifier(n_estimators=500, n_jobs = -1, random_state=0)
forest_clf.fit(X_train_sample, y_train_sample)
#
y_test_pred = forest_clf.predict(X_test)
X_test['y'] = y_test
X_test['y_pred'] = y_test_pred
#
true_index = np.where(X_test['y'] == True)[0]
false_index = np.where(X_test['y'] == False)[0]
#
X_test_true = X_test.iloc[true_index]
X_test_false = X_test.iloc[false_index]
X_test_true_raw = X_test_raw.iloc[true_index]
X_test_false_raw = X_test_raw.iloc[false_index]
correct_index = np.where(X_test_true['y'] == X_test_true['y_pred'])[0]
wrong_index = np.where(X_test_true['y'] != X_test_true['y_pred'])[0]
#
n_use = 23
X_test_correct = X_test_true.iloc[correct_index[:n_use]]
X_test_incorrect = X_test_true.iloc[wrong_index[:n_use]]
X_test_correct_raw = X_test_true_raw.iloc[correct_index[:n_use]]
X_test_incorrect_raw = X_test_true_raw.iloc[wrong_index[:n_use]]
X_test_temp = X_test_correct.append(X_test_incorrect)
X_test_temp_raw = X_test_correct_raw.append(X_test_incorrect_raw)
X_test_use = X_test_temp.append(X_test_false.iloc[:n_use*2])
X_test_use_raw = X_test_temp_raw.append(X_test_false_raw.iloc[:n_use*2])
#
#y_test = pd.DataFrame(y_test)
#y_test.columns = ['y']
pca = PCA(n_components=2)
train_features = list(X_test.columns)
train_features.remove('y')
train_features.remove('y_pred')
pca.fit(X_test[train_features])
pc_list = pca.transform(X_test_use[train_features])
X_test_use['PC1'] = pc_list[:,0]
X_test_use['PC2'] = pc_list[:,1]
#
X_test_use_raw.columns = [a+'_raw' for a in X_test_use_raw.columns]
#
for a in X_test_use_raw.columns:
    X_test_use[a] = X_test_use_raw[a].values
X_test_use.to_csv("X_test_use_2.csv", sep=',')
# save data to csv files
#X_train_sample.to_csv("X_train_sample.csv", sep=',')
#y_train_sample.to_csv("y_train_sample.csv", sep=',')
#X_test.to_csv("X_test.csv", sep=',')
#y_test.to_csv("y_test.csv", sep=',')

#==============================================================================
#                             Random Forest
#==============================================================================
# read in the data
#X_train_sample = pd.read_csv('X_train_sample.csv')
#y_train_sample = pd.read_csv('y_train_sample.csv')
#X_test = pd.read_csv('X_test.csv')
#y_test = pd.read_csv('y_test.csv')
#y_test.head(5)