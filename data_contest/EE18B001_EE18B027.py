#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Importing all necessary libraries

# General libraries/utilities
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn utilities
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, PowerTransformer
from sklearn.inspection import permutation_importance

# Bossting libraries
import lightgbm
from catboost import CatBoostRegressor
import xgboost as xgb

# **The `PRML MKN Jan-21 Dataset` folder containing all the *csv* files for the program should be in the path `dataset_path`**
dataset_path = 'prml-data'

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Feature engineering

# ## Setting matplotlib style

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style = 'darkgrid')
sns.set_palette('deep')


# ## Processing train.csv

# In[3]:


df_train = pd.read_csv(dataset_path+"/PRML MKN Jan-21 Dataset/train.csv")
print('Unique customers :', len(df_train['customer_id'].unique()))
print('Unique songs :', len(df_train['song_id'].unique()))
print('Number of NaN values :', df_train.isnull().sum().sum())
df_train.head()


# In[4]:


# Check for duplicates
df_train.duplicated().any()


# ### Getting average score of a song

# In[5]:


df_song_score = pd.DataFrame(df_train.groupby(['song_id'])['score'].mean())
df_song_score.rename(columns={'score': 'avg_song_score'}, inplace=True)
df_song_score.head()


# ### Getting average score given by a customer

# In[6]:


df_cust_score = pd.DataFrame(df_train.groupby(['customer_id'])['score'].mean())
df_cust_score.rename(columns={'score': 'avg_cust_score'}, inplace=True)
df_cust_score.head()


# ### Analysing song scores 

# In[7]:


df_train['score'].unique()


# In[8]:


# scores distribution
sns.kdeplot(df_train['score'], shade = True)
plt.title('Scores Distribution\n')
plt.xlabel('Score')
plt.ylabel('Frequency')


# ## Processing songs.csv

# In[9]:


df_songs = pd.read_csv(dataset_path+"/PRML MKN Jan-21 Dataset/songs.csv")
print('Languages :', df_songs['language'].unique())
print('Unique languages :', len(df_songs['language'].unique()))
df_songs.head()


# In[10]:


df_songs.info()


# ### Analyzing language column

# In[11]:


plt.style.use('ggplot')
df_songs.language.value_counts().plot.bar()


# In[12]:


# top 5 languages
df_songs['language'].value_counts().head(5).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()


# ### Analyzing released_year

# In[13]:


print('Minimum released year :', df_songs.released_year.min())
print('Maximum released year :', df_songs.released_year.max())
df_songs.released_year.hist(bins=100)


# ### Analyzing NaN values

# In[14]:


print('Number of NaN values in comments :', df_songs.number_of_comments.isnull().sum())
print('Number of NaN values in language :', df_songs['language'].isnull().sum())
print('Number of NaN values in years :', df_songs['released_year'].isnull().sum())


# ## Processing song_labels.csv

# In[15]:


df_labels = pd.read_csv(dataset_path+"/PRML MKN Jan-21 Dataset/song_labels.csv")
print('Unique labels :', len(df_labels['label_id'].unique()))
df_labels['label_id'].hist(bins=20)


# ### Binning label_id to 20 values

# In[16]:


s = pd.cut(df_labels['label_id'], bins=20)
mid_labels = np.array([int((a.left + a.right)/2.0) for a in s])
df_labels['label_id'] = mid_labels
df_labels.head()


# In[17]:


df_labels['label_id'].hist(bins=100)


# ### Analyzing platform_id

# In[18]:


print('Number of entries in song_labls.csv :', len(df_labels))
print('Unique platform_id :', len(df_labels['platform_id'].unique()))
df_labels.head()


# In[19]:


df_labels.info()


# ### Setting label with maximum count as the label for a song

# In[20]:


df_mxlabels = df_labels[df_labels.groupby(['platform_id'])['count'].transform(max) == df_labels['count']]
df_mxlabels.head()


# In[21]:


df_mxlabels = df_mxlabels.drop_duplicates(subset='platform_id', keep='first')
df_mxlabels.info()


# ## Processing save_for_later.csv

# In[22]:


df_save = pd.read_csv(dataset_path+"/PRML MKN Jan-21 Dataset/save_for_later.csv")
df_save.head()


# ### Getting song_save_counts (number of customer who saved a song)

# In[23]:


df_songsave = df_save.groupby(['song_id']).size().reset_index(name='song_save_counts')
df_songsave.info()


# ### Checking for new customers in save_for_later.csv not in train.csv

# In[24]:


df_save[~df_save['customer_id'].isin(df_train['customer_id'])].head()


# ### Creating feature vector for each customer in df_cust_final dataframe

# In[25]:


df_cust_final = pd.DataFrame({
    'customer_id': df_train['customer_id'].unique()
})
# cust_save_counts -- Number of songs saved by that customer
df_cust_final = pd.merge(df_cust_final, df_save.groupby(['customer_id']).size().reset_index(name='cust_save_counts'), 
                         how='left', on=['customer_id'])
df_cust_final.info()


# ### Filling NaN entries in cust_save_counts

# In[26]:


df_cust_final['cust_save_counts'].fillna(0, inplace=True)
df_cust_final['cust_save_counts'] = df_cust_final['cust_save_counts'].astype(np.int64)


# ### Creating feature vector for each song in df_song_final dataframe

# In[27]:


df_songs_final = pd.DataFrame({
    'song_id': list(range(1, 10001))
})
df_songs_final.head()


# ### Merging with features in df_songs

# In[28]:


df_songs_final = pd.merge(df_songs_final, df_songs, how='left', on=['song_id'])
df_songs_final.info()


# ### Considering released_year < 1500 as missing values (since they are unrealistic)

# In[29]:


df_songs_final.loc[df_songs_final['released_year'] < 1500, ['released_year']] = np.nan
df_songs_final.released_year.unique()


# ### Merging with df_mxlabels

# In[30]:


df_songs_final = pd.merge(df_songs_final, df_mxlabels, how='left', on=['platform_id'])
df_songs_final.head()


# ### Merging with df_songsave

# In[31]:


df_songs_final = pd.merge(df_songs_final, df_songsave, how='left', on=['song_id'])
df_songs_final.head()


# In[32]:


df_songs_final.info()


# ### Filling NaN values in label_id by observing those for the left out data

# In[33]:


df_mxlabels_left = df_mxlabels[~df_mxlabels['platform_id'].isin(df_songs_final['platform_id'])]
df_mxlabels_left.info()


# In[34]:


label_id_fill = np.bincount(df_mxlabels_left['label_id'].values).argmax()


# ### Filling NaN values in count with median because of discontinuous distribution

# In[35]:


df_mxlabels_left['count'].hist(bins=100)


# In[36]:


count_fill = int(df_mxlabels_left['count'].median())
count_fill


# In[37]:


df_mxlabels['count'].mean()


# ### Filling NaN values in number_of_comments with mean because of continuous distribution

# In[38]:


df_songs_final['number_of_comments'].hist(bins=100)


# In[39]:


no_comments_fill = int(df_songs_final['number_of_comments'].mean())
no_comments_fill


# ### Filling NaN values in song_save_counts with mean because of continuous distribution

# In[40]:


df_songsave['song_save_counts'].hist(bins=100)


# In[41]:


df_songsave['song_save_counts'].mean()


# ### Filling NaN values in released_year with mode of bins

# In[42]:


pd.cut(df_songs['released_year'], 100).value_counts().sort_index()


# In[43]:


released_yr_fill = int(pd.cut(df_songs['released_year'], 100).value_counts().sort_index().index[-1].mid)
released_yr_fill


# ### Bringing the lesser percentage categories and NaN in language to new label 'others'

# In[44]:


df_songs_final.loc[~df_songs_final['language'].isin(['eng', 'en-US', 'en-GB', 'en-CA', 'others']), ['language']] = 'others'
df_songs_final.language.unique()


# In[45]:


df_songs_final['label_id'].unique()


# ### NaN values filling in df_songs_final

# In[46]:


df_songs_final['label_id'].fillna(label_id_fill, inplace=True)
df_songs_final['label_id'] = df_songs_final['label_id'].astype(np.int64)
df_songs_final['count'].fillna(count_fill, inplace=True)
df_songs_final['count'] = df_songs_final['count'].astype(np.int64)
df_songs_final['number_of_comments'].fillna(no_comments_fill, inplace=True)
df_songs_final['number_of_comments'] = df_songs_final['number_of_comments'].astype(np.int64)
df_songs_final['song_save_counts'].fillna(0, inplace=True)
df_songs_final['song_save_counts'] = df_songs_final['song_save_counts'].astype(np.int64)
df_songs_final['released_year'].fillna(released_yr_fill, inplace=True)
df_songs_final['released_year'] = df_songs_final['released_year'].astype(np.int64)
df_songs_final['language'].fillna('others', inplace=True)

df_songs_final.info()


# ### Merging average scores with df_songs_final to get feature vector for songs

# #### Features definitions
# 
# song_id : ID of the song
# 
# platform_id : alternate ID for a song (will be ignore, not a part of the feature vector since it is incomplete)
# 
# released_year : year of release of song
# 
# language : language in which song is composed
# 
# number_of_comments : no of comments for the song
# 
# label_id : Most commonly used label for the song
# 
# count : number of people who gave that label for the song
# 
# song_save_counts : no of customers who saved the song to listen later
# 
# avg_song_score : average score the song received

# In[47]:


df_songs_final = pd.merge(df_songs_final, df_song_score, how='left', on=['song_id'])
df_songs_final.info()


# ### Merging df_save with most commonly used labels for the songs to get a customer's favourite label (genre)

# In[48]:


df_label_cust = pd.merge(df_songs_final[['song_id','label_id','count']], df_save, how='left', on=['song_id'])
df_label_cust.info()


# #### Finding sum of counts of same label corresponding to a customer 

# In[49]:


df_label_cust['count'] = df_label_cust.groupby(['customer_id','label_id'])['count'].transform('sum')
df_label_cust.head()


# In[50]:


df_label_cust.info()


# #### Function to get top k labels a customer is interested in

# In[51]:


def f(x, k=0):
    return list(df_label_cust.loc[x.index,['count','label_id']].sort_values('count').drop_duplicates('label_id').reset_index().loc[:k, 'label_id'])


# #### Removing same labels present for a customer after updating the counts

# In[52]:


df_label_cust.drop_duplicates(subset=['customer_id','label_id'], inplace=True)
df_label_cust.info()


# #### Get the the top k labels

# I use k = 1 since getting more label didn't make a difference in training

# In[53]:


df_tmp = pd.DataFrame(df_label_cust.groupby(['customer_id']).agg({'label_id': lambda x : f(x, 0)}))
df_tmp.head()


# #### Getting the most favourite label as top_1_label

# In[54]:


df_tmp[['top_1_label']] = pd.DataFrame(df_tmp['label_id'].to_list(), index=df_tmp.index)
df_tmp.head()


# ### Finding the maximum occuring top_1_label to fill NaNs

# In[55]:


top_1_fill = df_tmp.top_1_label.mode().iloc[0]
top_1_fill


# In[56]:


df_tmp['top_1_label'].fillna(top_1_fill, inplace=True)
df_tmp['top_1_label'] = df_tmp['top_1_label'].astype(np.int64)
df_tmp.head()


# ### Merging the favourite labels to df_cust_final

# In[57]:


df_cust_final = pd.merge(df_cust_final, df_tmp, how='left', on=['customer_id'])
df_cust_final.info()


# ### Filling NaNs in df_cust_final if any

# In[58]:


df_cust_final.drop(columns='label_id', inplace=True)
df_cust_final['top_1_label'].fillna(top_1_fill, inplace=True)
df_cust_final['top_1_label'] = df_cust_final['top_1_label'].astype(np.int64)
df_cust_final.info()


# ### Merging average scores given by a customer into df_cust_final

# In[59]:


df_cust_final = pd.merge(df_cust_final, df_cust_score, how='left', on=['customer_id'])
df_cust_final.info()


# In[60]:


df_songs_final.info()


# ### Features of song needed to be considered (song_id is a must as that is used to join with df_train and df_test)

# In[61]:


song_features_needed = ['song_id', 'number_of_comments', 'avg_song_score', 'released_year', 'count', 'language', 'label_id', 'song_save_counts']


# ### Merge song features into df_train

# In[62]:


df_train_final = pd.merge(df_train, df_songs_final[song_features_needed], how='left', on=['song_id'])
df_train_final.info()


# ### Features of customer needed to be considered (customer_id is a must as that is used to join with df_train and df_test)

# In[63]:


cust_features_needed = ['customer_id', 'cust_save_counts', 'avg_cust_score', 'top_1_label']


# ### Merging features of customers into df_train

# In[64]:


df_train_final = pd.merge(df_train_final, df_cust_final[cust_features_needed], how='left', on=['customer_id'])
df_train_final.info()


# ### List of categorical feature columns

# In[65]:


cat_cols = df_train_final.select_dtypes(exclude=['float64', 'int64']).columns
cat_cols = list(cat_cols) + [feature for feature in ['label_id', 'top_1_label', 'song_id'] 
                             if feature in song_features_needed or feature in cust_features_needed]
cat_cols


# ### Encoding object/string categorical features

# In[66]:


# encode customer ID
le_cust = preprocessing.LabelEncoder()
if 'customer_id' in cust_features_needed:
    df_train_final['customer_id'] = le_cust.fit_transform(df_train_final['customer_id'])


# In[67]:


# encode language
le_lang = preprocessing.LabelEncoder()
if 'language' in song_features_needed:
    # pd.get_dummies() to convert cat features with less labels to one hot, but didn't work well
    # enc_lang = pd.get_dummies(df_train_final['language'])
    # df_train_final = pd.concat([df_train_final, enc_lang], axis = 1)
    df_train_final['language'] = le_lang.fit_transform(df_train_final['language'])


# ### X : X_train | y : y_train

# In[68]:


X = df_train_final.drop(['score'], axis=1)
y = df_train_final['score']


# In[69]:


# split 80% of the data to the training set and 20% of the data to validation set 
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 2021)


# ### Visualizing the bivariate distribution

# In[70]:


sns.jointplot(x = 'score', y = 'avg_cust_score', data = df_train_final)


# ### Visualizing correlation plot

# In[71]:


plt.figure(figsize=(10,10))
sns.heatmap(df_train_final.corr(), annot=True, square=True, fmt='.2f')
plt.show()


# ### Getting count of each score

# In[72]:


np.bincount(df_train_final['score'])


# In[73]:


plt.scatter(x=np.arange(df_train_final.shape[0]),y=df_train_final['score'].sort_values())


# ### Merge song and customer features into df_test and do the necessary encoding of categorical features

# In[74]:


df_test = pd.read_csv(dataset_path+"/PRML MKN Jan-21 Dataset/test.csv")
X_test = pd.merge(df_test, df_songs_final[song_features_needed], how='left', on=['song_id'])
X_test = pd.merge(X_test, df_cust_final[cust_features_needed], how='left', on=['customer_id'])
if 'customer_id' in cust_features_needed:
    X_test['customer_id'] = le_cust.transform(X_test['customer_id'])
if 'language' in song_features_needed:
    # enc_lang = pd.get_dummies(X_test['language'])
    # X_test = pd.concat([X_test, enc_lang], axis = 1)
    X_test['language'] = le_lang.transform(X_test['language'])

X_test.info()


# ### Function to scale features in case of using normal regression methods

# In[75]:


def scale(scaler, X, X_test):
    scaler.fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    return X_scaled, X_test_scaled


# ### Defining the scaler

# In[76]:


scaler = RobustScaler()


# ### Including plynomial terms for the features (did not help much, so ignored)

# In[77]:


# poly = PolynomialFeatures(interaction_only=True).fit(X)
# X_poly = pd.DataFrame(poly.transform(X), columns=['feature_'+str(i) for i in range(poly.n_output_features_)], index=X.index)
# X_test_poly = pd.DataFrame(poly.transform(X_test), columns=['feature_'+str(i) for i in range(poly.n_output_features_)], index=X_test.index)


# In[78]:


X_scaled, X_test_scaled = scale(scaler, X, X_test)


# # Model selection and training

# ## LGBM

# We trained this model for several combination of hyperparameters and used this in ensemble with other models as well. But our final two submissions did not involve this model.

# ### Parameters to train the LGBM model

# In[79]:


lgbm_params = {
    'objective': 'regression', 
    'metric': 'mse', 
    'num_leaves': 45, 
    'max_depth': 8, 
    'lambda_l2': 1,
    'lambda_l1': 0.5,
    'num_threads': 10, 
    'learning_rate': 0.01, 
    # 'bagging_fraction': 0.9,
    # 'feature_fraction': 0.9,
    # 'bagging_freq': 5,
    'feature_fraction_seed': 42,
    'bagging_seed': 2021, 
    'verbosity': -1,
    # 'n_estimators': 300,
    # 'max_bin': 55
}


# ### Function to train the LGBM model

# In[80]:


def lgb_train(X, y, X_test, cat_cols, params):
    kf = StratifiedShuffleSplit(n_splits=5, random_state=2021)
    pred_test = 0
    pred_train = 0
    for train_index, val_index in kf.split(X, y):
        X_train, X_valid = X.iloc[train_index,:], X.iloc[val_index,:]
        y_train, y_valid = y[train_index], y[val_index]
        lgtrain = lightgbm.Dataset(X_train, y_train, categorical_feature=list(cat_cols))
        lgvalid = lightgbm.Dataset(X_valid, y_valid, categorical_feature=list(cat_cols))
        model = lightgbm.train(params, lgtrain, 2000, valid_sets=[lgvalid], early_stopping_rounds=100, verbose_eval=100)
        
        y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
        y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
        print('train mse:',metrics.mean_squared_error(y_train, y_pred_train),'| valid mse:',metrics.mean_squared_error(y_valid, y_pred_valid))

        pred_test_iter = model.predict(X_test, num_iteration=model.best_iteration)
        pred_test_iter[pred_test_iter<1]=1
        pred_test_iter[pred_test_iter>5]=5
        pred_test+=pred_test_iter
        pred_train_iter = model.predict(X, num_iteration=model.best_iteration)
        pred_train_iter[pred_train_iter<1]=1
        pred_train_iter[pred_train_iter>5]=5
        pred_train+=pred_train_iter
    pred_test /= 5.
    pred_train  /= 5.
    return pred_test, pred_train


# #### Features to ignore while training

# In[81]:


features_ignore = []


# #### Updated categorical columns after features are ignored

# In[82]:


cat_cols_2 = ['song_id', 'customer_id', 'label_id', 'top_1_label']


# ### Training the model

# In[83]:


# pred_test_lgb, pred_train_lgb = lgb_train(X.drop(columns=features_ignore), y, X_test.drop(columns=features_ignore), cat_cols_2, lgbm_params)


# ### MSE on overall training data

# In[84]:


# print('MSE:', metrics.mean_squared_error(y, pred_train_lgb))


# ## CatBoost

# This model performed much better than any other model in validation set. We used it in ensemble with Funk SVD solutions to get the best score on public leaderboard.

# ### Function to train the CatBoost model

# In[131]:


def cat_train(X, y, X_test, cat_cols):
    kf = StratifiedShuffleSplit(n_splits=5, random_state=2021)
    pred_test_cat = 0
    pred_train_cat = 0
    for train_index, val_index in kf.split(X, y):
        X_train, X_valid = X.loc[train_index,:], X.loc[val_index,:]
        y_train, y_valid = y[train_index], y[val_index]
        model = CatBoostRegressor(iterations=1500,
                                  use_best_model=True,
                                  learning_rate=0.05,
                                  depth=8,
                                  l2_leaf_reg=0.5,
                                  eval_metric='RMSE',
                                  random_seed = 2021,
                                  bagging_temperature = 0.8,
                                  od_type='Iter',
                                  metric_period = 100,
                                  od_wait=100)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid),use_best_model=True,verbose=True, 
                  cat_features= [i for i in range(len(X.columns)) if X.columns[i] in cat_cols])
        
        y_pred_train = model.predict(X_train)
        y_pred_valid = model.predict(X_valid)
        print('train mse:',metrics.mean_squared_error(y_train, y_pred_train),'| valid mse:',metrics.mean_squared_error(y_valid, y_pred_valid))

        pred_test = model.predict(X_test)
        pred_test[pred_test<1]=1
        pred_test[pred_test>5]=5
        pred_test_cat += pred_test
        pred_train = model.predict(X)
        pred_train[pred_train<1]=1
        pred_train[pred_train>5]=5
        pred_train_cat += pred_train
    pred_test_cat /= 5.
    pred_train_cat /= 5.
    return pred_test_cat, pred_train_cat


# #### Features to ignore while training

# In[132]:


features_ignore = ['language', 'top_1_label', 'label_id', 'count', 'cust_save_counts']


# #### Updated categorical columns after features are ignored

# In[133]:


cat_cols_2 = ['customer_id', 'song_id']


# ### Training the model

# In[134]:


pred_test_cat, pred_train_cat = cat_train(X.drop(columns=features_ignore), y, X_test.drop(columns=features_ignore), cat_cols_2)


# ### MSE on overall training data

# In[1]:


print('MSE:', metrics.mean_squared_error(y, pred_train_cat))


# ## XGBoost

# We also used this model with different hyperparameters and part of ensembles, but did not use it in final 2 submissions.

# ### Parameters to train the XGBoost model

# In[96]:


params_xgb = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mse',
    'eta': 0.05,
    # 'gamma': 1,
    'max_depth': 7,
    # 'n_estimators': 512,  
    'colsample_bytree': 0.7,
    # 'subsample': 0.7, 
    'lambda': 2,
    'random_state': 2021,
    'silent': True,
}


# ### Function to train the XGBoost model

# In[97]:


def xgb_train(X, y, X_test, params):
    kf = StratifiedShuffleSplit(n_splits=5, random_state=2021)
    pred_test_xgb = 0
    pred_train_xgb = 0
    for train_index, val_index in kf.split(X, y):
        X_train, X_valid = X.loc[train_index,:], X.loc[val_index,:]
        y_train, y_valid = y[train_index], y[val_index]
        xgb_train_data = xgb.DMatrix(X_train, y_train)
        xgb_val_data = xgb.DMatrix(X_valid, y_valid)
        xgb_submit_data = xgb.DMatrix(X_test)
        xgb_submit_data_train = xgb.DMatrix(X)
        xgb_model = xgb.train(params, xgb_train_data, 
                              num_boost_round=2000, 
                              evals= [(xgb_train_data, 'train'), (xgb_val_data, 'valid')],
                              early_stopping_rounds=100, 
                              verbose_eval=500
                             )
        
        y_pred_train = xgb_model.predict(xgb.DMatrix(X_train), ntree_limit=xgb_model.best_ntree_limit)
        y_pred_valid = xgb_model.predict(xgb.DMatrix(X_valid), ntree_limit=xgb_model.best_ntree_limit)
        print('train mse:',metrics.mean_squared_error(y_train, y_pred_train),'| valid mse:',metrics.mean_squared_error(y_valid, y_pred_valid))
        
        pred_test = xgb_model.predict(xgb_submit_data, ntree_limit=xgb_model.best_ntree_limit)
        pred_train = xgb_model.predict(xgb_submit_data_train, ntree_limit=xgb_model.best_ntree_limit)
        pred_test[pred_test<1]=1
        pred_test[pred_test>5]=5
        pred_train[pred_train<1.5]=1
        pred_train[pred_train>5]=5
        pred_test_xgb += pred_test
        pred_train_xgb += pred_train
        
    pred_test_xgb /= 5.
    pred_train_xgb /= 5.
    return pred_test_xgb, pred_train_xgb


# #### Features to ignore while training

# In[98]:


features_ignore = []


# ### Training the model

# In[99]:


# pred_test_xgb, pred_train_xgb = xgb_train(X.drop(columns=features_ignore), y, X_test.drop(columns=features_ignore), params_xgb)


# ### MSE on overall training data

# In[100]:


# print('MSE:', metrics.mean_squared_error(y, pred_train_xgb))


# ## Sklearn models

# Sklearn models like Lasso Regression , ENet Regression, Random Forest Regressor, Gradient Boosting Regressor also were attempted to use, but they did not perform as much as the other boosting libraries or Funk SVD.

# ### Function to train sklearn models

# In[ ]:


def sklearn_train(model, X, y, X_test):
    kf = StratifiedShuffleSplit(n_splits=5, random_state=2021)
    pred_test_model = 0
    pred_train_model = 0
    for train_index, val_index in kf.split(X, y):
        X_train, X_valid = X.loc[train_index,:], X.loc[val_index,:]
        y_train, y_valid = y[train_index], y[val_index]

        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_valid = model.predict(X_valid)
        print('train mse:',metrics.mean_squared_error(y_train, y_pred_train),'| valid mse:',metrics.mean_squared_error(y_valid, y_pred_valid))
        pred_test = model.predict(X_test)
        pred_train = model.predict(X)
        pred_test[pred_test<1]=1
        pred_test[pred_test>5]=5
        pred_train[pred_train<1]=1
        pred_train[pred_train>5]=5
        pred_test_model += pred_test
        pred_train_model += pred_train
        
    pred_test_model /= 5.
    pred_train_model /= 5.
    return pred_test_model, pred_train_model


# ## Lasso Regression

# ### Defining the model

# In[ ]:


# model_lasso = Lasso(alpha=0.05, random_state=2021)


# ### Learning and predicting

# In[ ]:


# pred_test_lasso, pred_train_lasso = sklearn_train(model_lasso, X_scaled, y, X_test_scaled)


# ### MSE on overall training data

# In[ ]:


# print('MSE:', metrics.mean_squared_error(y, pred_train_lasso))


# In[ ]:


# df_test_pred['score'] = (pred_test_lgb + pred_test_cat + pred_test_ENet + pred_test_KRR + pred_test_lasso) / 5.0
# df_test_pred.head(10)
# df_test_pred.to_csv('submit_8d.csv')


# ## Elastic Net Regression

# ### Defining the model

# In[ ]:


# model_ENet = ElasticNet(alpha=0.01, l1_ratio=.7, random_state=2021)


# ### Learning and predicting

# In[ ]:


# pred_test_ENet, pred_train_ENet = sklearn_train(model_ENet, X_scaled, y, X_test_scaled)


# ### MSE on overall training data

# In[ ]:


# print('MSE:', metrics.mean_squared_error(y, pred_train_ENet))


# ## Random Forest

# ### Defining the model

# In[102]:


# model_forest = RandomForestRegressor(max_depth=10, max_features=0.83, max_samples=0.7, random_state=2021)


# ### Learning and predicting

# In[103]:


# pred_test_forest, pred_train_forest = sklearn_train(model_forest, X, y, X_test)


# ### MSE on overall training data

# In[104]:


# print('MSE:', metrics.mean_squared_error(y, pred_train_forest))


# ### Finding importance of parameters for feature selection and engineering

# In[105]:


# result = permutation_importance(model_forest, X, y, n_repeats=10, random_state=42, n_jobs=2)


# In[106]:


# forest_importances = pd.Series(result.importances_mean, index=X.columns)

# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
# ax.set_title("Feature importances using permutation on full model")
# ax.set_ylabel("Mean accuracy decrease")
# fig.tight_layout()


# In[107]:


# result.importances_mean


# ## Gradient Bossting Regressor

# ### Defining the model

# In[108]:


# model_gbr = GradientBoostingRegressor(max_depth=8, subsample=0.7, max_features=0.9, random_state=2021)


# ### Learning and predicting

# In[109]:


# pred_test_gbr, pred_train_gbr = sklearn_train(model_gbr, X, y, X_test)


# ### MSE on overall training data

# In[110]:


# print('MSE:', metrics.mean_squared_error(y, pred_train_gbr))


# ## Funk SVD Matrix Factorisation (Latent factor model)

# In[111]:


get_ipython().run_line_magic('load_ext', 'Cython')


# ### Using Cython for the Funk SVD code alone

# In[112]:


get_ipython().run_cell_magic('cython', '', 'import time\nimport numpy as np\nimport pandas as pd\ncimport cython\ncimport numpy as np\n\ndef create_utility_matrix(data, field = {\'user\':0, \'item\': 1, \'score\': 2}):\n\n    """\n    Arguments:\n        data -- pandas dataframe, nx3\n        field -- dict having the column name or ids for users, items and score\n    Returns:    \n        1. 2D utility matrix (|U| x |S|, |U| = no of users, |S| = no of items) \n        2. list of users (in order with the utility matrix rows)\n        3. list of items (in order with the utility matrix columns)\n    """\n    item_col, user_col, score_col = field[\'item\'], field[\'user\'], field[\'score\']\n\n    userList, itemList, scoreList = data.loc[:,user_col].tolist(), data.loc[:,item_col].tolist(), \\\n                                    data.loc[:,score_col].tolist()\n\n    users = list(set(data.loc[:,user_col]))\n    items = list(set(data.loc[:,item_col]))\n\n    users_index = {users[i]: i for i in range(len(users))}\n    df_dict = {item: [np.nan for i in range(len(users))] for item in items}\n\n    for i in range(0,len(data)):\n        item, user, score = itemList[i], userList[i], scoreList[i]\n\n        df_dict[item][users_index[user]] = score\n\n    X = pd.DataFrame(df_dict)\n    X.index = users\n\n    users = list(X.index)\n    items = list(X.columns)\n\n    return np.array(X), users, items\n\n\n@cython.boundscheck(False)\n@cython.wraparound(False)\n\ncdef class FunkSVD:\n    cdef public int K\n    cdef public int n_iter\n    cdef public double lr\n    cdef public double reg\n    cdef public double reg_bias\n    cdef public double threshold\n    cdef public bint bias\n\n    cdef public double global_mean\n    cdef public list users, items\n    cdef public dict userdict, itemdict\n    cdef public np.ndarray userfeatures\n    cdef public np.ndarray itemfeatures\n    cdef public np.ndarray user_bias\n    cdef public np.ndarray item_bias\n    cdef public np.ndarray mask\n\n    def __init__(self, K=32, n_iter = 100, lr = 1e-5, reg = 1e-5, reg_bias = None, bias=True, threshold=1e-4):\n        self.K = K\n        self.n_iter = n_iter\n        self.lr = lr\n        self.reg = reg\n        self.bias = bias\n        self.reg_bias = reg if reg_bias is None else reg_bias\n        self.threshold = threshold\n\n    def fit(self, X, field={\'user\':0, \'item\':1, \'score\':2}, verbose = True):\n            self.stochasticGD(X, field, verbose)\n\n    def stochasticGD(self, X, field, verbose):\n        item_col, user_col, score_col = field[\'item\'], field[\'user\'], field[\'score\']\n        X = X[[user_col, item_col, score_col]]\n\n        cdef list users, items\n\n        users, items = list(set(X.loc[:, user_col])), list(set(X.loc[:, item_col]))\n\n        cdef double global_mean = np.mean(X.loc[:, score_col])\n        cdef double lr = self.lr\n        cdef double reg = self.reg\n        cdef double reg_bias = self.reg_bias\n        cdef np.ndarray[np.double_t, ndim=2] userfeatures = np.random.random((len(users), self.K))/self.K\n        cdef np.ndarray[np.double_t, ndim=2] itemfeatures = np.random.random((len(items), self.K))/self.K\n        cdef np.ndarray[np.double_t, ndim=2] user_bias = np.zeros((len(users), 1))\n        cdef np.ndarray[np.double_t, ndim=2] item_bias = np.zeros((1, len(items)))\n\n        cdef int i, f, useridx, itemidx\n        cdef double res, ui_dot, y_hat, r, error\n        cdef dict useridxes, itemidxes\n        cdef list ratings\n\n        useridxes, itemidxes = {users[i]:i for i in range(len(users))}, {items[i]:i for i in range(len(items))}\n        ratings = [(useridxes[x[0]], itemidxes[x[1]], x[2]) for x in X.values]\n        N = len(ratings)\n        \n        # For stopping when difference in cost function falls below the threshold \n        J_prev = 0\n\n        for epoch in range(self.n_iter):\n            error = 0\n            for useridx,itemidx,r in ratings:\n                res = 0\n                if self.bias == True:\n                    ui_dot = 0\n                    for f in range(self.K):\n                        ui_dot += userfeatures[useridx, f]*itemfeatures[itemidx, f]\n                    y_hat = ui_dot + global_mean + user_bias[useridx, 0] + item_bias[0, itemidx]\n                    res = r - y_hat\n                    user_bias[useridx, 0] += lr * (res - reg_bias * user_bias[useridx, 0])\n                    item_bias[0, itemidx] += lr * (res - reg_bias * item_bias[0, itemidx])\n                else:\n                    y_hat = 0\n                    for f in range(self.K):\n                        y_hat += userfeatures[useridx, f]*itemfeatures[itemidx, f]\n                    res = r - y_hat - global_mean\n\n                error += res**2\n\n                for f in range(self.K):\n                    userfeatures[useridx, f] += lr * (res * itemfeatures[itemidx, f] - reg * userfeatures[useridx, f])\n                    itemfeatures[itemidx, f] += lr * (res * userfeatures[useridx, f] - reg * itemfeatures[itemidx, f])\n\n            if verbose:\n                J = 0.5 * (error + reg_bias*np.sum(item_bias) + reg_bias*np.sum(user_bias) + \\\n                           reg*np.sum(itemfeatures) + reg*np.sum(userfeatures))\n                error = error / N\n                print("Epoch " + str(epoch) + ": Cost: " + str(J) + "| mse: " + str(error))\n            \n            if epoch > 0 and abs(J_prev - J) < self.threshold:\n                break\n            \n            J_prev = J\n\n        self.users = users\n        self.items = items\n        self.userfeatures = userfeatures\n        self.itemfeatures = itemfeatures\n        self.global_mean = global_mean\n        self.user_bias = user_bias\n        self.item_bias = item_bias\n        self.userdict = useridxes\n        self.itemdict = itemidxes\n\n    def predict(self, X, field = {\'user\': 0, \'item\': 1}, verbose=False):\n        """\n        Arguments:\n            X -- the test data set. 2D, array-like consisting of two columns\n                  corresponding to the user_id and item_id\n            field -- to provide a way of addressing the user_id and item_id\n        Return: \n            1D list giving the score corresponding to each row\n        """\n        cdef list testusers, testitems, users, items\n        cdef dict userdict, itemdict\n        cdef np.ndarray[np.double_t, ndim=2] userfeatures = self.userfeatures\n        cdef np.ndarray[np.double_t, ndim=2] itemfeatures = self.itemfeatures\n        cdef np.ndarray[np.double_t, ndim=2] user_bias = self.user_bias\n        cdef np.ndarray[np.double_t, ndim=2] item_bias = self.item_bias\n        cdef float global_mean = self.global_mean\n\n        testusers, testitems = X[field[\'user\']].tolist(), X[field[\'item\']].tolist()\n        users, items, userdict, itemdict = self.users, self.items, self.userdict, self.itemdict\n\n        # user and item in the test set may not always occur in the train set. \n        # So need to consider 4 cases :\n        #     1. both user and item in train\n        #     2. only user in train\n        #     3. only item in train\n        #     4. both not in train\n\n        cdef list predictions\n        cdef int useridx, itemidx\n        predictions = []\n\n        start_time = time.clock()\n\n        for i in range(len(testusers)):\n            user = testusers[i]\n            item = testitems[i]\n            if user in userdict and item in itemdict:\n                useridx = userdict[user]\n                itemidx = itemdict[item]\n                ssum = np.sum(userfeatures[useridx] * itemfeatures[itemidx]) + global_mean\n                pred = ssum + user_bias[useridx, 0] + item_bias[0, itemidx]\n                \n            elif user in userdict:\n                useridx = userdict[user]\n                if self.bias:\n                    predictions = global_mean + user_bias[useridx, 0]\n                else:\n                    pred = global_mean + np.sum(userfeatures[useridx] * np.mean(itemfeatures, axis=0))\n                    \n            elif item in itemdict:\n                itemidx = itemdict[item]\n                if self.bias:\n                    pred = global_mean + item_bias[0, itemidx]\n                else:\n                    pred = global_mean + np.sum(itemfeatures[itemidx] * np.mean(userfeatures, axis=0))\n                    \n            else:\n                pred = global_mean\n\n            predictions.append(pred)\n\n        print("time taken {} secs".format(time.clock() - start_time))\n\n        return np.array(predictions)')


# ### Splitting into train and validation set to analyse model

# In[113]:


X_train_f, X_valid_f, y_train_f, y_valid_f = train_test_split(df_train[['customer_id', 'song_id']], 
                                                              df_train['score'], 
                                                              test_size=0.2, random_state=42)


# ### First model (with quite a large number of hidden features) - K = 100

# #### Training model on the train set

# In[114]:


f1 = FunkSVD(K=100, lr=0.005, reg=0.08, reg_bias=0.06, n_iter=200, bias=True)

# fits the model to the data
f1.fit(X=pd.concat([X_train_f, y_train_f], axis=1), field={'user':'customer_id', 'item':'song_id', 'score':'score'}, verbose=True)


# #### Prediction and MSE on train, valid and overall data

# In[115]:


pred_train_f1 = f1.predict(X=X_train_f, field={'user':'customer_id', 'item':'song_id'})
valid_pred_f1 = f1.predict(X=X_valid_f, field={'user':'customer_id', 'item':'song_id'})
pred_f1 = f1.predict(X=df_train[['customer_id', 'song_id']], field={'user':'customer_id', 'item':'song_id'})

print('Train MSE :', metrics.mean_squared_error(pred_train_f1, y_train_f))
print('Valid MSE :', metrics.mean_squared_error(valid_pred_f1, y_valid_f))
print('Overall MSE :', metrics.mean_squared_error(pred_f1, y))


# #### Training model on whole dataset

# In[116]:


f1_final = FunkSVD(K=100, lr=0.005, reg=0.08, reg_bias=0.06, n_iter=200, bias=True)

# fits the model to the data
f1_final.fit(X=df_train, field={'user':'customer_id', 'item':'song_id', 'score':'score'}, verbose=True)


# #### Prediction on train and test data

# In[117]:


pred_train_f1 = f1_final.predict(X=df_train[['customer_id', 'song_id']], 
                                 field={'user':'customer_id', 'item':'song_id'})
pred_test_f1 = f1_final.predict(X=df_test, field={'user':'customer_id', 'item':'song_id'})


# ### Second model (with a small number of hidden features) - K = 25
# 
# This was used for the second submission

# #### Training model on the train set

# In[118]:


# f2 = FunkSVD(K=25, lr=0.005, reg=0.14, reg_bias=0.08, n_iter=200, bias=True)

# # fits the model to the data
# f2.fit(X=pd.concat([X_train_f, y_train_f], axis=1), field={'user':'customer_id', 'item':'song_id', 'score':'score'}, verbose=True)


# #### Prediction and MSE on train, valid and overall data

# In[119]:


# pred_train_f2 = f2.predict(X=X_train_f, field={'user':'customer_id', 'item':'song_id'})
# valid_pred_f2 = f2.predict(X=X_valid_f, field={'user':'customer_id', 'item':'song_id'})
# pred_f2 = f2.predict(X=df_train[['customer_id', 'song_id']], field={'user':'customer_id', 'item':'song_id'})

# print('Train MSE :', metrics.mean_squared_error(pred_train_f2, y_train_f))
# print('Valid MSE :', metrics.mean_squared_error(valid_pred_f2, y_valid_f))
# print('Overall MSE :', metrics.mean_squared_error(pred_f2, y))


# #### Training model on whole dataset

# In[120]:


# f2_final = FunkSVD(K=25, lr=0.005, reg=0.14, reg_bias=0.08, n_iter=200, bias=True)

# # fits the model to the data
# f2_final.fit(X=df_train, field={'user':'customer_id', 'item':'song_id', 'score':'score'}, verbose=True)


# #### Prediction on train and test data

# In[121]:


# pred_train_f2 = f2_final.predict(X=df_train[['customer_id', 'song_id']], 
#                                  field={'user':'customer_id', 'item':'song_id'})
# pred_test_f2 = f2_final.predict(X=df_test, field={'user':'customer_id', 'item':'song_id'})


# # Ensemble

# Collaborative filtering techniques overfit a lot more than regression and boosting solutions since they fill the missing values based on the existing ones. But they usually perform well on test set also and has been the winning model in many cases. It is also used by a lot of companies. We wanted to ensemble this overfitting model with a well regularised catboost solution and use.
# 
# Our first submission is the one which performed best on public leaderboard, consisting of two overfitting Funk SVD solutions (these two are mentioned in the code) and a well regularised catboost solution (with parameters mentioned in code)
# 
# Our second submission is a slightly more regularised Funk SVD configuration (K=25, reg=0.14, reg_bias=0.08, n_iters=200, lr=0.005). This gave a public leaderboard score of 0.72989 which is pretty good and also was more regularised than the top solution, so we felt might work well on private leaderboard.
# 
# For catboost (and other boosting and regression methods) we used cross validation to train the model on entire dataset.
# For Funk SVD we trained on a randomly chosen 80% percent of the data and used the 20% as validation data to tune the hyperparameters. Then for submitting we trained the same model using the whole data and used it to predict the test set scores.

# In[122]:


df_test_pred = pd.DataFrame({'score': pred_test_cat.tolist()})
df_test_pred.index.name = 'test_row_id'
df_test_pred.head()


# #### First submission

# In[123]:


pred_test_overall = (pred_test_cat + pred_test_f1) / 2.0
pred_train_overall = (pred_train_cat + pred_train_f1) / 2.0


# #### Second submission

# In[ ]:


# pred_test_overall_2 = (pred_test_cat + pred_test_f2) / 2.0
# pred_train_overall_2 = (pred_train_cat + pred_train_f2) / 2.0


# ### Ensemble model MSE on overall train data

# In[124]:


print('MSE:', metrics.mean_squared_error(y, pred_train_overall))


# ### Writing ensemble model prediction to df_test_pred

# In[125]:


df_test_pred['score'] = pred_test_overall
df_test_pred.head(10)


# ### Getting the predictions file

# In[126]:


df_test_pred.to_csv('submit_final_1.csv')

