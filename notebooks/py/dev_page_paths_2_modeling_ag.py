
# coding: utf-8

# %load jupyter_default.py
import pandas as pd
import numpy as np
import os
import re
import datetime
import time
import glob
import json
from tqdm import tqdm_notebook, tqdm
from colorama import Fore, Style

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from slugify import slugify

get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
sns.set() # Revert to matplotlib defaults
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelpad'] = 20
plt.rcParams['legend.fancybox'] = True
plt.style.use('ggplot')

SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 16, 20
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)

def savefig(name):
    plt.savefig(f'../../figures/{name}.png', bbox_inches='tight', dpi=300)

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
    
get_ipython().run_line_magic('reload_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'pandas, numpy')


from google.cloud import bigquery


from dotenv import load_dotenv
load_dotenv('../../.env')


client = bigquery.Client()


# # Page Paths
# Alex's development notebook for page paths.

# ## Read data

# ### Read precomputed local file

def load_file(f_path):
    if not os.path.exists(f_path):
        print('No data found. Run data load script above.')
        return
    print('Loading {}'.format(f_path))
    df = pd.read_csv(f_path)
    df.date = pd.to_datetime(df.date)
    return df

df_long = load_file('../../data/interim/page_paths_raw.csv')


df_long.head()


# df['week'] = df.date.apply(lambda x: x.strftime('%W'))
# df['year'] = df.date.apply(lambda x: x.strftime('%Y'))
# df['week_start'] = df[['week', 'year']].apply(
#     lambda x: datetime.datetime.strptime('{}-{}-1'.format(x.year, x.week), '%Y-%W-%w'),
#     axis=1
# )


df_long.dtypes


# ### Build session aggregate dataframe

CELL_TRIGGER = True

def make_session_agg_df(df) -> pd.DataFrame:
    df_ = df.copy()
    df_.transactions = df_.transactions.fillna(0)

    data = []
    cols = ['id', 'date', 'is_mobile', 'transactions', 'num_hits',
            'page_paths', 'page_titles']

    for id_ in tqdm_notebook(df_['id'].drop_duplicates().tolist()):
        m = df_['id'] == id_
        s = df_[m]
        data.append([
            id_,
            s.date.iloc[0],
            s.isMobile.iloc[0],
            s.transactions.sum(),
            s.numHits.iloc[0],
            s.pagePath.tolist(),
            s.pageTitle.tolist(),
        ])

    return pd.DataFrame(data, columns=cols)

if CELL_TRIGGER:
    df_sess = make_session_agg_df(df_long)
    df_sess.to_csv('../../data/interim/page_paths_sess_agg_raw.csv', index=False)
    df_sess.to_pickle('../../data/interim/page_paths_sess_agg_raw.pkl')


df_sess.head()


df_sess.tail()


df_sess.dtypes


# ### Load session aggregate dataframe

CELL_TRIGGER = True

def load_file(f_path, load_pkl):
    if not os.path.exists(f_path):
        print('No data found. Run data load script above.')
        return
    print('Loading {}'.format(f_path))
    if load_pkl:
        df = pd.read_pickle(f_path)
    else:
        df = pd.read_csv(f_path)
        df.date = pd.to_datetime(df.date)
    return df

if CELL_TRIGGER:
    df_sess = load_file('../../data/interim/page_paths_sess_agg_raw.pkl', load_pkl=True)


df_sess.head()


df_sess.tail()


df_sess.dtypes


# Fixing bug where page title is NaN for the homepage...
# 
# e.g.

df_sess.iloc[35915]


# **NOTE: making a huge assumption here (that ever NaN is a homepage - might not be the case). Be more carful if running this in production.**

df_sess['page_titles'] = df_sess['page_titles'].apply(lambda seq: ['Home' if str(x) == 'nan' else x for x in seq])


# ## Data mining

from prefixspan import PrefixSpan
from sklearn.preprocessing import LabelEncoder


# Setting up sequential pattern mining

import hashlib
hash_sha224 = lambda x: hashlib.sha224(x.encode('utf-8')).hexdigest()

def custom_filter(patt, matches=None):
    """
    Filter top sequence results, removing matches that
        - Have length 1
        - Have duplicate elements
        
    patt : list
        Sequential pattern e.g. [1, 2]
        
    matches : list
        Index and position of matches (not used).
        e.g. [(0, 1), (1, 0), (2, 1), (3, 0)]
    """
    patt_len = len(patt)
    if patt_len <= 1:
        return False
    elif len(set(patt)) != patt_len:
        return False
    return True


def callback(patt, matches, top_k, patt_indices, label_encoder):
    """
    Save index results using this callback
    
    patt, matches : lists
        Passed by PrefixSpan.topk
        
    top_k : list
        Store counts and page_labels
        
    patt_indices : dict
        Store training data indices for feature label keys

    """
    pattern = label_encoder.inverse_transform(patt)
    patt_id = hash_sha224('_'.join(pattern))
    patt_indices[patt_id] = [m[0] for m in matches]
    top_k.append([len(matches), pattern])


def get_topk_sequences(sequences, label_encoder, k, custom_filter, callback):
    labeled_sequences = [label_encoder.transform(seq) for seq in sequences]
    prefix_spans = PrefixSpan(labeled_sequences)
    patt_indices = {}
    top_k = []
    prefix_spans.topk(k, callback=lambda patt, matches: callback(patt, matches, top_k, patt_indices, label_encoder))
    
    out = []
    for count, page_labels in top_k:
        if custom_filter(page_labels):
            out.append([count, page_labels])

    return out, patt_indices


# Setting up label encoders

page_paths = list(df_long.pagePath.unique())
page_path_le = LabelEncoder()
page_path_le.fit(page_paths)

page_titles = list(df_long.pageTitle.unique())
page_title_le = LabelEncoder()
page_title_le.fit(page_titles)


# Let's focus on page titles for now. Can loop back and look at page path later

# for i, seq in enumerate(sequences):
#     try:
#         page_title_le.transform(seq)
#     except Exception as e:
#         print(e)
#         print(i)
#         print(seq)


import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


sequences = df_sess.page_titles.tolist()

topk_raw, topk_patt_indices_raw = get_topk_sequences(
    sequences,
    label_encoder=page_title_le,
    k=1000,
    custom_filter=custom_filter,
    callback=callback,
)


topk_raw[:50]


print('Indeces for top patterns:')

patt_eg = topk_raw[0][1]
patt_id = hash_sha224('_'.join(patt_eg))
print(patt_eg)
print(topk_patt_indices_raw[patt_id])
print()

patt_eg = topk_raw[1][1]
patt_id = hash_sha224('_'.join(patt_eg))
print(patt_eg)
print(len(topk_patt_indices_raw[patt_id]))


# Dump calculation to file

CELL_TRIGGER = True

if CELL_TRIGGER:
    
    # Save the topk paths
    with open('../../data/interim/path_paths_topk_1000_titles.jsonl', 'w') as f:
        for line in topk_raw:
            f.write('{}\n'.format(
                json.dumps({'count': line[0], 'pattern': list(line[1])})
            ))
            
    # Load the index lookup dict
    with open('../../data/interim/page_paths_topk_1000_titles_patt_indices.json', 'w') as f:
        f.write(json.dumps(topk_patt_indices_raw))


# Add as wide-form (one hot encoded) features to `df_sess`

# ### Modling converting / non-converting sessions with page path features

# #### Read in topk page paths

CELL_TRIGGER = True

if CELL_TRIGGER:
    
    # Load the topk paths
    topk = []
    with open('../../data/interim/path_paths_topk_1000_titles.jsonl', 'r') as f:
        topk = [json.loads(line) for line in f.read().splitlines() if line.strip()]
        
    # Load the index lookup dict
    with open('../../data/interim/page_paths_topk_1000_titles_patt_indices.json', 'r') as f:
        topk_patt_indices = json.loads(f.read())
    


print(json.dumps(topk_patt_indices, indent=4)[:500])


import hashlib
hash_sha224 = lambda x: hashlib.sha224(x.encode('utf-8')).hexdigest()


df_topk = pd.DataFrame(topk)
df_topk['pattern_str'] = df_topk.pattern.astype(str)
df_topk['pattern_hash'] = df_topk.pattern.apply(lambda x: hash_sha224('_'.join(x)))
df_topk.head()


# Add pattern ID to map indices

top_k_pattern_hashes = df_topk[['pattern_hash', 'pattern_str']].values.tolist()


top_k_pattern_hashes[:5]


# dict(
#     list(topk_counts.items())[:10]
# )


# Label training data

df_training = pd.read_pickle('../../data/interim/page_paths_sess_agg_raw.pkl')


df_training.head()


# Fill NaN page_paths with `Home`.

# **NOTE: making a huge assumption here (that ever NaN is a homepage - might not be the case). Be more carful if running this in production.**

df_training['page_titles_nan_fill'] = df_training['page_titles'].apply(
    lambda seq: ['Home' if str(x) == 'nan' else x for x in seq]
)


import hashlib
hash_sha224 = lambda x: hashlib.sha224(x.encode('utf-8')).hexdigest()
df_training['page_titles_hash'] = df_training..apply(
    lambda x: hash_sha224('_'.join(x))
)


df_training.head()


# Attach features

for seq_hash, seq_name in top_k_pattern_hashes:
    del df_training[seq_name]


for seq_hash, seq_name in top_k_pattern_hashes:
    df_training[seq_hash] = 0
    m = df_training.index.isin(topk_patt_indices.get(seq_hash, []))
    if m.sum() == 0:
        print('WARNING: no matches in training data for {}'.format(seq_name))
    df_training.loc[m, seq_hash] = 1


# Dump results (after long computation)
df_training.to_csv('../../data/interim/page_paths_k_1000_titles_training_data.csv', index=False)
df_training.to_pickle('../../data/interim/page_paths_k_1000_titles_training_data.pkl')


df_training.head()


# Adding labels for converting / non-converting

df_training['converting_session'] = (df_training.transactions > 0).astype(int)
df_training['converting_session'].isnull().sum()


df_training['converting_session'].value_counts()


df_training.columns[-10:]


non_features = ['converting_session', 'id', 'date', 'is_mobile', 'transactions',
                'num_hits', 'page_paths', 'page_titles', 'page_titles_nan_fill', 
                'page_titles_hash']
target = 'converting_session'
features = [col for col in df_training.columns if col not in non_features]

print('{} non-feature colums ({}...)\n{} feature columns ({}...)\ntarget={}'.format(
    len(non_features), ', '.join(non_features[:3]),
    len(features), ', '.join(features[:3]),
    target
))


# Modeling with a decision tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve


# Use a scoring function that's sensitive to false negatives (e.g. F1 score)

get_ipython().run_line_magic('pinfo', 'validation_curve')


get_ipython().run_line_magic('pinfo', 'RandomForestClassifier')


get_ipython().run_cell_magic('time', '', "\nX = df_training[features].values\ny = df_training[target].values\n\nclf = RandomForestClassifier(n_estimators=10)\nmax_depths = np.logspace(1, 3, 5)\n\ntrain_scores, test_scores = validation_curve(\n            estimator=clf,\n            X=X,\n            y=y,\n            param_name='max_depth',\n            param_range=max_depths,\n            cv=5,\n            verbose=10,\n            scoring='f1',\n);")


# Function to draw the validation curve

def plot_validation_curve(train_scores, test_scores,
                          param_range, xlabel='', log=False):
    '''
    This code is from scikit-learn docs:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    
    Also here:
    https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/code/ch06/ch06.ipynb
    '''
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig = plt.figure()
    
    plt.plot(param_range, train_mean, 
             color=sns.color_palette('Set1')[1], marker='o', 
             markersize=5, label='training accuracy')

    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color=sns.color_palette('Set1')[1])

    plt.plot(param_range, test_mean, 
             color=sns.color_palette('Set1')[0], linestyle='--', 
             marker='s', markersize=5, 
             label='validation accuracy')

    plt.fill_between(param_range, 
                     test_mean + test_std,
                     test_mean - test_std, 
                     alpha=0.15, color=sns.color_palette('Set1')[0])

    if log:
        plt.xscale('log')
    plt.legend(loc='lower right')
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel('Accuracy')
    plt.ylim(0.9, 1.0)
    return fig


plot_validation_curve(train_scores, test_scores,
                      max_depths, xlabel='max_depth');


get_ipython().run_cell_magic('time', '', "\nX = df_training[features].values\ny = df_training[target].values\n\nclf = RandomForestClassifier(n_estimators=50)\nmax_depths = np.linspace(1, 30, 5)\n\ntrain_scores, test_scores = validation_curve(\n            estimator=clf,\n            X=X,\n            y=y,\n            param_name='max_depth',\n            param_range=max_depths,\n            cv=5,\n            verbose=10,\n            scoring='f1',\n);")


plot_validation_curve(
    train_scores, test_scores,
    max_depths, xlabel='max_depth'
);


# It looks like max depth of 10 is good.
# 
# I'm not looking to do a deep dive into model selection or parameter tuning. I am more interested in training a model with good bias variance tradeoff and looking at the feature importances.

get_ipython().run_cell_magic('time', '', '\nX = df_training[features].sample(frac=1).values\ny = df_training[target].sample(frac=1).values\n\nclf = RandomForestClassifier(max_depth=10, n_estimators=100)\nclf.fit(X, y)')


from sklearn.metrics import accuracy_score, f1_score, classification_report 


y_hat = clf.predict(X)


print(classification_report(y, y_hat))


# Support just lists the value counts.

np.unique(y, return_counts=True)[1]


# I do not understand why there are multiple f1-scores. Also, how can precision be so high while recall is zero?

get_ipython().run_line_magic('pinfo', 'f1_score')


f1_score(y, y_hat)


# 
#     ``'binary'``:
#         Only report results for the class specified by ``pos_label``.
#         This is applicable only if targets (``y_{true,pred}``) are binary.
#     ``'micro'``:
#         Calculate metrics globally by counting the total true positives,
#         false negatives and false positives.
#     ``'macro'``:
#         Calculate metrics for each label, and find their unweighted
#         mean.  This does not take label imbalance into account.
#     ``'weighted'``:
#         Calculate metrics for each label, and find their average, weighted
#         by support (the number of true instances for each label). This
#         alters 'macro' to account for label imbalance; it can result in an
#         F-score that is not between precision and recall.
#     ``'samples'``:
#         Calculate metrics for each instance, and find their average (only
#         meaningful for multilabel classification where this differs from
#         :func:`accuracy_score`).
# 

f1_score(y, y_hat, average='binary', pos_label=0)


f1_score(y, y_hat, average='binary', pos_label=1)


f1_score(y, y_hat, average='micro')


f1_score(y, y_hat, average='macro')


f1_score(y, y_hat, average='weighted')


# I think recall is zero because no class was given a label of 1. Let's check the confusion matrix for class probabilities

from sklearn.metrics import confusion_matrix


y.shape


confusion_matrix(y, y_hat)


cmat = confusion_matrix(y, y_hat)
cmat.diagonal() / cmat.sum(axis=1) * 100


# This is brutal. We only predicted 4/11548 right for the class of interest.
# 
# I want to do the hyperparameter tuning with a better f-score metric. Trying to explitily tell it to use `binary`. I think it's doing weighted by default.
# 
# AH! The problem before was that I didn't shuffle the samples around.

get_ipython().run_line_magic('pinfo', 'make_scorer')


get_ipython().run_cell_magic('time', '', "\nX = df_training[features].sample(frac=1).values\ny = df_training[target].sample(frac=1).values\n\n\nf1_binary_scorer = make_scorer(f1_score, average='binary', pos_label=1)\n\nclf_ = RandomForestClassifier(n_estimators=50)\nmax_depths = np.linspace(1, 30, 5)\n\ntrain_scores, test_scores = validation_curve(\n            estimator=clf_,\n            X=X,\n            y=y,\n            param_name='max_depth',\n            param_range=max_depths,\n            cv=5,\n            verbose=10,\n            scoring=f1_binary_scorer,\n);")


train_scores


test_scores


plot_validation_curve(
    train_scores, test_scores,
    max_depths, xlabel='max_depth'
);


# Let's try using a custom scorer so we know exactly what should be happening

def custom_score(y_true, y_pred):
    """
    Return the accuracy for class 1.
    """
    m = y_true == 1
    return y_pred[m].sum() / m.sum()

custom_scorer = make_scorer(custom_score)


get_ipython().run_cell_magic('time', '', '\nX = df_training[features].sample(frac=0.1).values\ny = df_training[target].sample(frac=0.1).values\n\nclf = RandomForestClassifier(max_depth=10, n_estimators=10)\nclf.fit(X, y)')


custom_score(y, clf.predict(X))


get_ipython().run_cell_magic('time', '', "\nX = df_training[features].sample(frac=1).values\ny = df_training[target].sample(frac=1).values\n\nclf_ = RandomForestClassifier(n_estimators=10)\nmax_depths = np.linspace(1, 30, 5)\n\ntrain_scores, test_scores = validation_curve(\n            estimator=clf_,\n            X=X,\n            y=y,\n            param_name='max_depth',\n            param_range=max_depths,\n            cv=5,\n            verbose=10,\n            scoring=custom_scorer,\n);")


# Since all I care about is finding patterns for converting segments, let's remove any sequences that do not exist for converting segments.
# 
# It will also be interesting to look at the paths that get removed, along with their frequency 

print('Make sure that only page path hashes are below...')
print(df_training[features].columns)


m = df_training.transactions > 0
# todo remove head()
s = df_training[m][features].sum()
page_path_hash_non_converting = s[s == 0].index.tolist()


page_path_hash_non_converting = list(page_path_hash_non_converting)


# OK so there are no patterns that are unique to non-converters. Let's again try out our decision tree parameter search - this time over a wider range of max depth options.

# Function to draw the validation curve

def plot_validation_curve(train_scores, test_scores,
                          param_range, xlabel='', log=False):
    '''
    This code is from scikit-learn docs:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    
    Also here:
    https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/code/ch06/ch06.ipynb
    '''
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig = plt.figure()
    
    plt.plot(param_range, train_mean, 
             color=sns.color_palette('Set1')[1], marker='o', 
             markersize=5, label='training accuracy')

    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color=sns.color_palette('Set1')[1])

    plt.plot(param_range, test_mean, 
             color=sns.color_palette('Set1')[0], linestyle='--', 
             marker='s', markersize=5, 
             label='validation accuracy')

    plt.fill_between(param_range, 
                     test_mean + test_std,
                     test_mean - test_std, 
                     alpha=0.15, color=sns.color_palette('Set1')[0])

    if log:
        plt.xscale('log')
    plt.legend(loc='lower right')
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel('Score')
#     plt.ylim(0.9, 1.0)
    return fig


get_ipython().run_cell_magic('time', '', "\nX = df_training[features].sample(frac=1).values\ny = df_training[target].sample(frac=1).values\n\nclf_ = RandomForestClassifier(n_estimators=100)\nmax_depths = np.linspace(1, 100, 10)\n\ntrain_scores, test_scores = validation_curve(\n            estimator=clf_,\n            X=X,\n            y=y,\n            param_name='max_depth',\n            param_range=max_depths,\n            cv=5,\n            verbose=10,\n            scoring=make_scorer(f1_score, average='binary', pos_label=1),\n);")


plot_validation_curve(
    train_scores, test_scores,
    max_depths, xlabel='max_depth'
);


# ### Page sequence trends over time

# ### Top page transitions
# 
# Given a specific page, what's the probability of going to each other page? Where does site exits rank in here.

# ### TODO
# 
# - Add DATA_PATH global variable to Gdrive. Move `interim` files
# 
# 
# - Predict purchase intent with decision tree. Look at top features. Feature engineering ideas:
#     - Split out into converting and non-converting segments
#     - Split out into mobile vs desktop
# 
# 
# - See how page sequence trends change over time
# - Given a specific page, what's the most common next page(s)?

from IPython.display import HTML
HTML('<style>div.text_cell_render{font-size:130%;padding-top:50px;padding-bottom:50px}</style>')

