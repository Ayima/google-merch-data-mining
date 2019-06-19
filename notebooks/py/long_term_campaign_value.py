
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


# _Note: source data is ~900k total lines_

# # Long Term Campaign Value

# ## Read data

# Since there's no userID in the Google Merch Store dataset, we'll assign random user IDs to each sessions.

# ### Read from `jsonl` (simulating `user_id`)

n = 1e5
p = 0.9
simulated_user_ids = np.random.binomial(n, p, int(9e5))
simulated_user_ids = np.abs(simulated_user_ids - simulated_user_ids.mean())
plt.hist(simulated_user_ids)


len(np.unique(simulated_user_ids))


width = 1e4
len(np.unique((np.random.exponential(scale=1, size=int(9e5)) * width).astype(int)))


get_ipython().run_cell_magic('time', '', '"""\nUsing local jsonl\n"""\nERRORS = []\n\ndef pull_daily_data(verbose=False, raise_errors=False):\n    dataset = sorted(glob.glob(\'../../data/raw/*.jsonl\'))\n    \n    # Set random seed for simulated user ID\n    np.random.seed(19)\n    # Set user dilution. Higher means more users\n    user_dilution = 1e5\n    \n    data = []\n    user_id_dates = {}\n    user_id_campaigns = {}\n    for table in tqdm_notebook(dataset):\n        if verbose:\n            print(\'Scanning {}\'.format(table))\n        with open(table, \'r\') as f:\n            table_data = []\n            for line in f:\n                try:\n                    d = json.loads(line)\n                    simulated_user_id = int(np.random.exponential(scale=1) * user_dilution)\n                    \n                    if d[\'trafficSource\'][\'campaign\'] != \'(not set)\':\n                        new_acquisition = True\n                        date_acquired = d[\'date\']\n                        campaign_acquired = float(\'nan\')\n                        table_data.append([\n                            d[\'date\'],\n                            d[\'userId\'],\n                            simulated_user_id,\n                            d[\'trafficSource\'][\'campaign\'],\n                            d[\'trafficSource\'][\'source\'],\n                            d[\'trafficSource\'][\'medium\'],\n                            d[\'totals\'][\'transactions\'],\n                            d[\'totals\'][\'totalTransactionRevenue\'],\n                            new_acquisition,\n                            date_acquired,\n                            campaign_acquired,\n                        ])\n                        user_id_dates[simulated_user_id] = d[\'date\']\n                        user_id_campaigns[simulated_user_id] = d[\'trafficSource\'][\'campaign\']\n\n                    elif simulated_user_id in user_id_dates.keys():\n                        new_acquisition = False\n                        date_acquired = user_id_dates[simulated_user_id]\n                        campaign_acquired = user_id_campaigns.get(simulated_user_id, float(\'nan\'))\n                        table_data.append([\n                            d[\'date\'],\n                            d[\'userId\'],\n                            simulated_user_id,\n                            d[\'trafficSource\'][\'campaign\'],\n                            d[\'trafficSource\'][\'source\'],\n                            d[\'trafficSource\'][\'medium\'],\n                            d[\'totals\'][\'transactions\'],\n                            d[\'totals\'][\'totalTransactionRevenue\'],\n                            new_acquisition,\n                            date_acquired,\n                            campaign_acquired,\n                        ])\n\n                except Exception as e:\n                    if verbose:\n                        print(\'Error raised when reading row:\\n{}\'.format(e))\n                    ERRORS.append([table, e])\n                    if raise_errors:\n                        raise(e)\n\n            cols = [\'date\', \'userId\', \'simulated_user_id\', \'campaign\', \'source\', \'medium\',\n                    \'transactions\', \'totalTransactionRevenue\', \'new_acquisition\', \'date_acquired\',\n                   \'campaign_acquired\']\n            if table_data:\n                results = pd.DataFrame(table_data, columns=cols)\n                data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df')


# Cancel this execution after a few files have been parsed without error
jsonl_results = pull_daily_data(raise_errors=True, verbose=True)


jsonl_results = pull_daily_data()


len(ERRORS), ERRORS[:10]


df = jsonl_results.copy()
df.date = pd.to_datetime(df.date)
df.date_acquired = pd.to_datetime(df.date_acquired)
df['first_acquisition_time_delta'] = df.date - df.date_acquired
df.to_csv('../../data/interim/long_term_campaign_value_simulated_userid_raw.csv', index=False)


def load_file(f_path):
    if not os.path.exists(f_path):
        print('No data found. Run data load script above.')
        return
    print('Loading {}'.format(f_path))
    df = pd.read_csv(f_path)
    df.date = pd.to_datetime(df.date)
    df.date_acquired = pd.to_datetime(df.date_acquired)
    df['first_acquisition_time_delta'] = df.date - df.date_acquired
    return df

# Looking forward to walrus operator for stuff like this...
tmp = load_file('../../data/interim/long_term_campaign_value_simulated_userid_raw.csv')
if tmp is not None:
    print('Loading from file')
    df = tmp.copy()
    del tmp


df.head()


df.dtypes


len(df)


(df.new_acquisition==False).sum()


df.userId.isnull().sum() / len(df)


df.simulated_user_id.value_counts().head()


df[df.simulated_user_id==5652]


# ### Plotting results

df[df.new_acquisition == False].first_acquisition_time_delta.astype('timedelta64[D]').plot.hist()
plt.xlabel('Days since campaign acquisition')
plt.ylabel('Number of users')


m = ~(df.new_acquisition)
s = df[m].first_acquisition_time_delta.astype('timedelta64[D]')
s = s[s>0]
s.value_counts().sort_index().cumsum().plot()
plt.xlabel('Days since user acquisition')
plt.ylabel('Number of cumulative sessions')


df_ = df.copy()
m = (~(df_.new_acquisition)) & (~(df_.transactions.isnull()))
df_.first_acquisition_time_delta = df_.first_acquisition_time_delta.astype('timedelta64[D]')
df_[m].groupby('first_acquisition_time_delta').transactions.sum().cumsum().plot()
plt.xlabel('Days since user acquisition')
plt.ylabel('Number of cumulative transactions')
del df_


df_ = df.copy()
m = (~(df_.new_acquisition)) & (~(df_.totalTransactionRevenue.isnull()))
df_.first_acquisition_time_delta = df_.first_acquisition_time_delta.astype('timedelta64[D]')
(df_[m].groupby('first_acquisition_time_delta').totalTransactionRevenue.sum().cumsum() / 1e6).plot()
plt.xlabel('Days since user acquisition')
plt.ylabel('Cumulative transaction revnue (USD)')
del df_


top_campaigns = df.groupby('campaign').totalTransactionRevenue.sum().sort_values(ascending=False).index.tolist()


top_campaigns


top_campaigns = df[m].groupby('campaign_acquired').totalTransactionRevenue.sum().sort_values(ascending=False).index.tolist()


top_campaigns


# Hard code campaigns from above into call below

def plot_campaign_long_term_value(df, what, campaign, color=None):
    m_campaign = df.campaign_acquired == campaign
    num_samples = (m & m_campaign).sum()
    print('Found {} samples for campaign: {}'.format(num_samples, campaign))
    if num_samples == 0:
        print('Skipping')
        return

    plot_args = {'label': campaign}
    if color is not None:
        plot_args['color'] = color
    
    (df[m & m_campaign].groupby('first_acquisition_time_delta')[what].sum().cumsum() / 1e6)        .plot(**plot_args)
    plt.xlabel('Days since user acquisition')
    plt.ylabel('Cumulative transaction revnue (USD)')


df_ = df.copy()
# Filter on previously acquired users who made a transaction
m = (~(df_.new_acquisition)) & (~(df_.totalTransactionRevenue.isnull()))
df_.first_acquisition_time_delta = df_.first_acquisition_time_delta.astype('timedelta64[D]')

top_campaigns = [
    'Data Share Promo',
    'AW - Dynamic Search Ads Whole Site',
    'AW - Accessories',
    'AW - Electronics',
    'All Products'
]

for campaign in top_campaigns:
    plot_campaign_long_term_value(df_, what='totalTransactionRevenue', campaign=campaign)
    
plt.legend()
plt.xlabel('Days since user acquisition')
plt.ylabel('Cumulative transaction revnue (USD)')
plt.show()
del df_


from IPython.display import HTML
HTML('<style>div.text_cell_render{font-size:130%;padding-top:50px;padding-bottom:50px}</style>')

