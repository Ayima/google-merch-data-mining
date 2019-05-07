
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
from tqdm import tqdm_notebook
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

# Look for some samples...

# ## Read data

# ### Read from bigquery (old code)

get_ipython().run_cell_magic('time', '', '"""\nUsing bigquery\n"""\n\ndef pull_daily_data(verbose=False):\n    dataset = client.get_dataset(\'bigquery-public-data.google_analytics_sample\')\n\n    data = []\n    for table in tqdm_notebook(list(client.list_tables(dataset))):\n        if verbose:\n            print(\'Querying {}\'.format(table.table_id))\n        query_job = client.query(\'\'\'\n            select\n              date,\n              sum(totals.visits),\n              sum(totals.pageviews),\n              sum(totals.transactions),\n              sum(totals.transactionRevenue)\n            from `bigquery-public-data.google_analytics_sample.{}`\n            group by date;\n        \'\'\'.format(table.table_id))\n        results = query_job.result().to_dataframe()\n        results.columns = [\'date\', \'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']\n        data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df\n\nbq_results = pull_daily_data()')


bq_results.head()


# ### Read `jsonl` from google drive (old code)

get_ipython().run_cell_magic('time', '', '"""\nUsing gdrive jsonl\n"""\nERRORS = []\n\ndef pull_daily_data(verbose=False, raise_errors=False):\n    dataset = sorted(glob.glob(\'/Volumes/GoogleDrive/My Drive/bigquery_ga_sample/*.jsonl\'))\n\n    data = []\n    for table in tqdm_notebook(dataset):\n        if verbose:\n            print(\'Scanning {}\'.format(table))\n        with open(table, \'r\') as f:\n            table_data = []\n            for line in f:\n                d = json.loads(line)\n                date = d[\'date\']\n                d = d[\'totals\']\n                try:\n                    table_data.append([\n                        date,\n                        d[\'visits\'],\n                        d[\'pageviews\'],\n                        d[\'transactions\'],\n                        d[\'transactionRevenue\'],\n                    ])\n                except Exception as e:\n                    if verbose:\n                        print(\'Error raised when reading row:\\n{}\'.format(e))\n                    ERRORS.append([table, e])\n                    if raise_errors:\n                        raise(e)\n\n            cols = [\'date\', \'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']\n            results = (\n                pd.DataFrame(table_data, columns=cols)\n                    .groupby(\'date\')[[\'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']]\n                    .sum().reset_index()\n            )\n            data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df\n\njsonl_gdrive_results = pull_daily_data()')


jsonl_gdrive_results


# ### Read `jsonl` from local

get_ipython().run_cell_magic('time', '', '"""\nUsing local jsonl\n"""\nERRORS = []\n\ndef pull_daily_data(verbose=False, raise_errors=False):\n    dataset = sorted(glob.glob(\'../../data/raw/*.jsonl\'))\n\n    data = []\n    i = 0\n    for table in tqdm_notebook(dataset):\n        if verbose:\n            print(\'Scanning {}\'.format(table))\n        with open(table, \'r\') as f:\n            table_data = []\n            visitor_id_dates = {}\n            for line in f:\n                i += 1\n                try:\n                    d = json.loads(line)\n                    d_visit = [\n                        i,\n                        d[\'date\'],\n                        d[\'device\'][\'isMobile\'],\n                        d[\'totals\'][\'transactions\'],\n                        len(d[\'hits\']),\n                    ]\n                    for h in d[\'hits\']:\n                        table_data.append(d_visit + [\n                            h[\'hitNumber\'],\n                            h[\'page\'][\'pagePath\'],\n                            h[\'page\'][\'pageTitle\'],\n                        ])\n\n                except Exception as e:\n                    if verbose:\n                        print(\'Error raised when reading row:\\n{}\'.format(e))\n                    ERRORS.append([table, e])\n                    if raise_errors:\n                        raise(e)\n\n            cols = [\'id\', \'date\', \'isMobile\', \'transactions\', \'numHits\',\n                    \'hitNumber\', \'pagePath\', \'pageTitle\',]\n            results = pd.DataFrame(table_data, columns=cols)\n            data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df')


jsonl_results = pull_daily_data(raise_errors=True, verbose=True)


jsonl_results = pull_daily_data()


df = jsonl_results.copy()
df.date = pd.to_datetime(df.date)
df.to_csv('../../data/interim/page_paths_raw.csv', index=False)


def load_file(f_path):
    if not os.path.exists(f_path):
        print('No data found. Run data load script above.')
        return
    print('Loading {}'.format(f_path))
    df = pd.read_csv(f_path)
    df.date = pd.to_datetime(df.date)
    return df

# Looking forward to walrus operator for stuff like this...
tmp = load_file('../../data/interim/page_paths_raw.csv')
if tmp is not None:
    print('Loading from file')
    df = tmp.copy()
    del tmp


df.head()


# df['week'] = df.date.apply(lambda x: x.strftime('%W'))
# df['year'] = df.date.apply(lambda x: x.strftime('%Y'))
# df['week_start'] = df[['week', 'year']].apply(
#     lambda x: datetime.datetime.strptime('{}-{}-1'.format(x.year, x.week), '%Y-%W-%w'),
#     axis=1
# )


df.dtypes


# ## Exploring the dataset

df.date.isnull().sum()


df.date.value_counts().sort_index().plot()


df.isMobile.value_counts()


# Number of unique sessions

len(df['id'].unique())


# Top landing pages:

df[df.hitNumber == 1].pagePath.value_counts(ascending=False)[:10]


# Top exit pages:

df[df.hitNumber == df.numHits].pagePath.value_counts(ascending=False)[:10]


# ## Pattern mining
# 
# Finding sequantial patterns (of pages).
# 
# Below we use the [`prefix span`](https://github.com/chuanconggao/PrefixSpan-py) python library.
# 
# e.g.
# ```python
# 
# from prefixspan import PrefixSpan
# 
# db = [
#     [0, 1, 2, 3, 4],
#     [1, 1, 1, 3, 4],
#     [2, 1, 2, 2, 0],
#     [1, 1, 1, 2, 2],
# ]
# 
# ps = PrefixSpan(db)
# 
# print(ps.frequent(2))
# # [(2, [0]),
# #  (4, [1]),
# #  (3, [1, 2]),
# #  (2, [1, 2, 2]),
# #  (2, [1, 3]),
# #  (2, [1, 3, 4]),
# #  (2, [1, 4]),
# #  (2, [1, 1]),
# #  (2, [1, 1, 1]),
# #  (3, [2]),
# #  (2, [2, 2]),
# #  (2, [3]),
# #  (2, [3, 4]),
# #  (2, [4])]
# 
# ```

from prefixspan import PrefixSpan


from sklearn.preprocessing import LabelEncoder


get_ipython().run_line_magic('pinfo', 'LabelEncoder')


page_le = LabelEncoder()
df['page_label'] = page_le.fit_transform(df.pagePath.values)


df.head()


def make_page_sequences(df) -> list:
    data, d = [], []
    prev_id = df.iloc[0]['id']
    for _, row in tqdm_notebook(df.iterrows(), total=len(df)):
        if prev_id != row['id']:
            data.append(d)
            d = []
        d.append(row['page_label'])
        prev_id = row['id']

    if d:
        data.append(d)

    return data

page_sequences = make_page_sequences(df)


prefix_spans = PrefixSpan(page_sequences)


prefix_spans.topk(10)


def get_topk_sequences(page_sequences, k):
    prefix_spans = PrefixSpan(page_sequences)
    top_k = prefix_spans.topk(k)
    out = []
    for count, page_labels in top_k:
        out.append([count, [page_le.inverse_transform(x) for x in page_labels]])

    return out

get_topk_sequences(page_sequences, 10)


# Let's drop all the rows where consecutive hits are from the same page

def make_page_sequences(df) -> list:
    data, d = [], []
    prev_id = df.iloc[0]['id']
    prev_page = None
    for _, row in tqdm_notebook(df.iterrows(), total=len(df)):
        if prev_id != row['id']:
            if len(d) > 1:
                data.append(d)
            d = []
            prev_page = None
        
        if (prev_page is None) or (row['page_label'] != prev_page):
            d.append(row['page_label'])

        prev_id = row['id']
        prev_page = row['page_label']
        
    if d:
        data.append(d)

    return data

page_sequences = make_page_sequences(df)


for i, d in enumerate(page_sequences):
    prev = -1
    for d_ in d:
        if d_ == prev:
            print(i)
        prev = d_


page_sequences[:10]


def get_topk_sequences(page_sequences, k, ignore_singles=True):
    prefix_spans = PrefixSpan(page_sequences)
    top_k = prefix_spans.topk(k)
    out = []
    for count, page_labels in top_k:
        if ignore_singles and (len(page_labels) == 1):
            continue
        out.append([count, [page_le.inverse_transform(x) for x in page_labels]])
        
    return out

get_topk_sequences(page_sequences, 10)


# Ah.. the issue here is that is that pages can be skipped. For example "home" -> "pdp" -> "home" can result in the extraction of the first pattern above :(

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

