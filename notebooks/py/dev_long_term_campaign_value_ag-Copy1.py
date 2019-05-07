
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


# # Long Term Campaign Value
# Alex's development notebook for long term campaign value.

# Look for some samples...

def search():
    data = []
    i = 0
    for f_name in glob.glob('../../data/raw/*.jsonl'):
        print('scanning {}'.format(f_name))
        with open(f_name, 'r') as f:
            for line in tqdm_notebook(f.readlines()):
                i += 1
                d = json.loads(line.strip())
                if d['trafficSource']['campaign'] != '(not set)':
                    data.append(d)
                    return data

print(json.dumps(search()))

# with open('/tmp/temp.json', 'w') as f:
#     json.dump(data, f)


def search():
    data = []
    i = 0
    for f_name in glob.glob('../../data/raw/*.jsonl'):
        print('scanning {}'.format(f_name))
        with open(f_name, 'r') as f:
            for line in tqdm_notebook(f.readlines()):
                i += 1
                d = json.loads(line.strip())
                if d['userId'] is not None:
                    print(d['userId'])
                    print(type(d['userId']))
                    print('Found one!')
                    data.append(d)
                    return data
#                 if d['trafficSource']['campaign'] != '(not set)':
#                     data.append(d)
#                     return data

print(json.dumps(search()))

# with open('/tmp/temp.json', 'w') as f:
#     json.dump(data, f)


# OK so there's no user ID in this dataset.
# 
# What is `fullVisitorId`

def search():
    data = []
    i = 0
    for f_name in glob.glob('../../data/raw/*.jsonl'):
        print('scanning {}'.format(f_name))
        with open(f_name, 'r') as f:
            for line in tqdm_notebook(f.readlines()):
                i += 1
                print(i)
                d = json.loads(line.strip())
                if d['fullVisitorId'] in data:
                    print('Repeat {}'.format(d['fullVisitorId']))
                    return {}
                data.append(d['fullVisitorId'])
                

print(json.dumps(search()))

# with open('/tmp/temp.json', 'w') as f:
#     json.dump(data, f)


# Let's use this as a unique ID

# ## Read data

# ### Read from bigquery (old code)

get_ipython().run_cell_magic('time', '', '"""\nUsing bigquery\n"""\n\ndef pull_daily_data(verbose=False):\n    dataset = client.get_dataset(\'bigquery-public-data.google_analytics_sample\')\n\n    data = []\n    for table in tqdm_notebook(list(client.list_tables(dataset))):\n        if verbose:\n            print(\'Querying {}\'.format(table.table_id))\n        query_job = client.query(\'\'\'\n            select\n              date,\n              sum(totals.visits),\n              sum(totals.pageviews),\n              sum(totals.transactions),\n              sum(totals.transactionRevenue)\n            from `bigquery-public-data.google_analytics_sample.{}`\n            group by date;\n        \'\'\'.format(table.table_id))\n        results = query_job.result().to_dataframe()\n        results.columns = [\'date\', \'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']\n        data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df\n\nbq_results = pull_daily_data()')


bq_results.head()


# ### Read `jsonl` from google drive (old code)

get_ipython().run_cell_magic('time', '', '"""\nUsing gdrive jsonl\n"""\nERRORS = []\n\ndef pull_daily_data(verbose=False, raise_errors=False):\n    dataset = sorted(glob.glob(\'/Volumes/GoogleDrive/My Drive/bigquery_ga_sample/*.jsonl\'))\n\n    data = []\n    for table in tqdm_notebook(dataset):\n        if verbose:\n            print(\'Scanning {}\'.format(table))\n        with open(table, \'r\') as f:\n            table_data = []\n            for line in f:\n                d = json.loads(line)\n                date = d[\'date\']\n                d = d[\'totals\']\n                try:\n                    table_data.append([\n                        date,\n                        d[\'visits\'],\n                        d[\'pageviews\'],\n                        d[\'transactions\'],\n                        d[\'transactionRevenue\'],\n                    ])\n                except Exception as e:\n                    if verbose:\n                        print(\'Error raised when reading row:\\n{}\'.format(e))\n                    ERRORS.append([table, e])\n                    if raise_errors:\n                        raise(e)\n\n            cols = [\'date\', \'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']\n            results = (\n                pd.DataFrame(table_data, columns=cols)\n                    .groupby(\'date\')[[\'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']]\n                    .sum().reset_index()\n            )\n            data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df\n\njsonl_gdrive_results = pull_daily_data()')


jsonl_gdrive_results


# ### Read `jsonl` from local

get_ipython().run_cell_magic('time', '', '"""\nUsing local jsonl\n"""\nERRORS = []\n\ndef pull_daily_data(verbose=False, raise_errors=False):\n    dataset = sorted(glob.glob(\'../../data/raw/*.jsonl\'))\n\n    data = []\n    for table in tqdm_notebook(dataset):\n        if verbose:\n            print(\'Scanning {}\'.format(table))\n        with open(table, \'r\') as f:\n            table_data = []\n            visitor_id_dates = {}\n            for line in f:\n                try:\n                    d = json.loads(line)\n                    \n                    if d[\'trafficSource\'][\'campaign\'] != \'(not set)\':\n                        new_acquisition = True\n                        date_acquired = float(\'nan\')\n                        table_data.append([\n                            d[\'date\'],\n                            d[\'userId\'],\n                            d[\'fullVisitorId\'],\n                            d[\'trafficSource\'][\'campaign\'],\n                            d[\'trafficSource\'][\'source\'],\n                            d[\'trafficSource\'][\'medium\'],\n                            d[\'transactionRevenue\'],\n                            new_acquisition,\n                            date_acquired,\n                        ])\n                        visitor_id_dates[d[\'fullVisitorId\']] = d[\'date\']\n                        \n                    elif d[\'fullVisitorId\'] in visitor_id_dates.keys():\n                        new_acquisition = False\n                        date_acquired = visitor_id_dates[d[\'fullVisitorId\']]\n                        table_data.append([\n                            d[\'date\'],\n                            d[\'userId\'],\n                            d[\'fullVisitorId\'],\n                            d[\'trafficSource\'][\'campaign\'],\n                            d[\'trafficSource\'][\'source\'],\n                            d[\'trafficSource\'][\'medium\'],\n                            d[\'transactionRevenue\'],\n                            new_acquisition,\n                            date_acquired,\n                        ])\n                        \n                except Exception as e:\n                    if verbose:\n                        print(\'Error raised when reading row:\\n{}\'.format(e))\n                    ERRORS.append([table, e])\n                    if raise_errors:\n                        raise(e)\n\n            cols = [\'date\', \'userId\', \'fullVisitorId\', \'campaign\', \'source\', \'medium\', \'transactionRevenue\', \'new_acquisition\', \'date_acquired\']\n            results = pd.DataFrame(table_data, columns=cols)\n            data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df\n\njsonl_results = pull_daily_data()')


df = jsonl_results.copy()
df.date = pd.to_datetime(df.date)
df.to_csv('../../data/interim/long_term_campaign_value_raw.csv', index=False)


def load_file(f_path):
    if not os.path.exists(f_path):
        print('No data found. Run data load script above.')
        return
    print('Loading {}'.format(f_path))
    df = pd.read_csv(f_path)
    df.date = pd.to_datetime(df.date)
    return df

# Looking forward to walrus operator for stuff like this...
tmp = load_file('../../data/interim/long_term_campaign_value_raw.csv')
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


from IPython.display import HTML
HTML('<style>div.text_cell_render{font-size:130%;padding-top:50px;padding-bottom:50px}</style>')

