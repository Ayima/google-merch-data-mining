
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

def savefig(plt, name):
    plt.savefig(f'../../figures/{name}.png', bbox_inches='tight', dpi=300)

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
    
get_ipython().run_line_magic('reload_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'pandas, numpy')


# # GA Data Mining
# Alex's development notebook for miscellaneous rough work.

# ## Misc
# 
# Note:
# > The first 1 TB of query data processed per month is free
# 
# I've downloaded it all as JSON, but I'm going to use bigquery SQL until I run out of calls

from google.cloud import bigquery


from dotenv import load_dotenv
load_dotenv('../../.env')


client = bigquery.Client()


query_job = client.query("""
    SELECT
        *
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20170801`
    LIMIT 10
""")

results = query_job.result()  # Waits for job to complete.


data = [r for r in results]


# Here is one row of data

dict(data[0].items())


json.dumps(dict(data[0].items()))


# We got 10 total rows

# Let's get some info about the dataset

dataset_id = 'bigquery-public-data.google_analytics_sample'
dataset = client.get_dataset(dataset_id)


full_dataset_id = "{}.{}".format(dataset.project, dataset.dataset_id)
friendly_name = dataset.friendly_name
print(
    "Got dataset '{}' with friendly_name '{}'.".format(
        full_dataset_id, friendly_name
    )
)

# View dataset properties
print("Description: {}".format(dataset.description))
print("Labels:")
labels = dataset.labels
if labels:
    for label, value in labels.items():
        print("\t{}: {}".format(label, value))
else:
    print("\tDataset has no labels defined.")

# View tables in dataset
print("Tables:")
tables = list(client.list_tables(dataset))  # API request(s)
if tables:
    for table in tables:
        print("\t{}".format(table.table_id))
else:
    print("\tThis dataset does not contain any tables.")


# Export the data

def dump_jsonl(big_query_results, f_path):
    """
    Feed in big query result, e.g.
    
    >>> query_job = client.query('''
    >>>     SELECT *
    >>>     FROM `bigquery-public-data.google_analytics_sample.{}`
    >>>     LIMIT 10
    >>> '''.format(table_id))
    >>> results = query_job.result()  # Waits for job to complete.
    >>> dump_jsonl(results, '../../data/raw/{}.jsonl'.format(table_id))
    
    """
    print('Writing {} lines to {}'.format(len(big_query_results), f_path))
    with open(f_path, 'w') as f:
        for result in results:
            f.write('{}\n'.format(
                json.dumps(dict(big_query_results.items()))
            ))


# ## Data Mining

# ### Predicting Product Revenue

table_id = 'ga_sessions_20160801'
query_job = client.query('''
    select date, sum(totals.totalTransactionRevenue)
    from `bigquery-public-data.google_analytics_sample.{}`
    group by date
    limit 10;
'''.format(table_id))
results = query_job.result()


dict(results)


get_ipython().run_cell_magic('time', '', '"""\nUsing bigquery\n"""\n\ndef pull_daily_data(verbose=False):\n    dataset = client.get_dataset(\'bigquery-public-data.google_analytics_sample\')\n\n    data = []\n    for table in tqdm_notebook(list(client.list_tables(dataset))):\n        if verbose:\n            print(\'Querying {}\'.format(table.table_id))\n        query_job = client.query(\'\'\'\n            select\n              date,\n              sum(totals.visits),\n              sum(totals.pageviews),\n              sum(totals.transactions),\n              sum(totals.transactionRevenue)\n            from `bigquery-public-data.google_analytics_sample.{}`\n            group by date;\n        \'\'\'.format(table.table_id))\n        results = query_job.result().to_dataframe()\n        results.columns = [\'date\', \'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']\n        data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df\n\nbq_results = pull_daily_data()')


bq_results.head()


get_ipython().run_cell_magic('time', '', '"""\nUsing local jsonl\n"""\nERRORS = []\n\ndef pull_daily_data(verbose=False, raise_errors=False):\n    dataset = sorted(glob.glob(\'../../data/raw/*.jsonl\'))\n\n    data = []\n    for table in tqdm_notebook(dataset):\n        if verbose:\n            print(\'Scanning {}\'.format(table))\n        with open(table, \'r\') as f:\n            table_data = []\n            for line in f:\n                d = json.loads(line)\n                date = d[\'date\']\n                d = d[\'totals\']\n                try:\n                    table_data.append([\n                        date,\n                        d[\'visits\'],\n                        d[\'pageviews\'],\n                        d[\'transactions\'],\n                        d[\'transactionRevenue\'],\n                    ])\n                except Exception as e:\n                    if verbose:\n                        print(\'Error raised when reading row:\\n{}\'.format(e))\n                    ERRORS.append([table, e])\n                    if raise_errors:\n                        raise(e)\n\n            cols = [\'date\', \'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']\n            results = (\n                pd.DataFrame(table_data, columns=cols)\n                    .groupby(\'date\')[[\'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']]\n                    .sum().reset_index()\n            )\n            data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df\n\njsonl_results = pull_daily_data()')


jsonl_results.head()


from IPython.display import HTML
HTML('<style>div.text_cell_render{font-size:130%;padding-top:50px;padding-bottom:50px}</style>')

