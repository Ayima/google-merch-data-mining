
# coding: utf-8

import pandas as pd
import numpy as np
import os
import re
import datetime
import time
import glob
import json
from tqdm import tqdm, tqdm_notebook
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

def savefig(name):
    plt.savefig(f'../../results/figures/{name}.png', bbox_inches='tight', dpi=300)

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
    
get_ipython().run_line_magic('reload_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'pandas, numpy')


from dotenv import load_dotenv
load_dotenv('../../.env')


# # Association Rules

# Searching for products that are commonly purchased together.

# ## Read data

# ### Read from BigQuery
# 
# Same code as `sales_forecast.ipynb`.

get_ipython().run_cell_magic('time', '', '"""\nUsing bigquery\n"""\n\nfrom google.cloud import bigquery\n# Using environment variables to authenticate\nclient = bigquery.Client()\n\ndef pull_daily_product_sales(verbose=False):\n    dataset = client.get_dataset(\'bigquery-public-data.google_analytics_sample\')\n\n    data = []\n    for table in tqdm_notebook(list(client.list_tables(dataset))):\n        if verbose:\n            print(\'Querying {}\'.format(table.table_id))\n        query_job = client.query(\'\'\'\n        SELECT\n            h.item.productName AS other_purchased_products,\n            COUNT(h.item.productName) AS quantity\n        FROM `bigquery-public-data.google_analytics_sample.{}`,\n            UNNEST(hits) as h\n        WHERE (\n            fullVisitorId IN (\n                SELECT fullVisitorId\n                FROM `bigquery-public-data.google_analytics_sample.{}`,\n                    UNNEST(hits) as h\n                WHERE h.item.productName CONTAINS \'Product Item Name A\'\n                AND totals.transactions>=1\n                GROUP BY fullVisitorId\n            )\n            AND h.item.productName IS NOT NULL\n            AND h.item.productName != \'Product Item Name A\'\n        )\n        GROUP BY other_purchased_products\n        ORDER BY quantity DESC;\n        \'\'\'.format(table.table_id, table.table_id))\n        results = query_job.result().to_dataframe()\n        results.columns = [\'date\', \'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']\n        data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df\n\nbq_product_results = pull_daily_product_sales()')


bq_product_results.head()


df = bq_product_results.copy()
df.date = pd.to_datetime(df.date)

f_path = os.path.join(os.getenv('DATA_PATH'), 'interim', 'sales_forecast_by_product_raw.csv')
if os.path.isfile(f_path):
    raise Exception(
        'File exists! Run line below in separate cell to overwrite it. '
        'Otherwise just run cell below to load file.')

df.to_csv(f_path, index=False)


# ### Load pre-queried data

def load_file(f_path):
    if not os.path.exists(f_path):
        print('No data found. Run data load script above.')
        return
    print('Loading {}'.format(f_path))
    df = pd.read_csv(f_path)
    df.date = pd.to_datetime(df.date)
    return df

f_path = os.path.join(os.getenv('DATA_PATH'), 'interim', 'sales_forecast_by_product_raw.csv')
df = load_file(f_path)


df.head()


# [source -> support.google.com](https://support.google.com/analytics/answer/3437719?hl=en)
# 
# > **visitId**   
# An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.

df['visitId'].nunique()


df['fullVisitorId'].nunique()


df['transactionId'] = df['visitId'].astype(str) + '|' + df['fullVisitorId'].astype(str)
df['transactionId'].nunique()


df['fullProductName'] = ''

# Do not include product variants for "Single Option Only"
m = df['productVariant'] == 'Single Option Only'
df.loc[m, 'fullProductName'] = df.loc[m, 'v2ProductName'].str.strip()
df.loc[~m, 'fullProductName'] = (
    df.loc[~m, 'v2ProductName'].str.strip()
    + ' ('
    + df.loc[~m, 'productVariant'].str.strip()
    + ')'
)


df_itemsets_by_transaction = df.groupby('transactionId')['fullProductName'].apply(list).to_frame()


df_itemsets_by_user = df.groupby('fullVisitorId')['fullProductName'].apply(list).to_frame()


pd.options.display.max_colwidth = 1000


# Some transactions with multiple items:

df_itemsets_by_transaction[df_itemsets_by_transaction.fullProductName.apply(lambda x: len(x) > 1)].head()


# Set up non-variant dataset as well

df_itemsets_by_transaction_no_variants = df.groupby('transactionId')['v2ProductName'].apply(list).to_frame()
df_itemsets_by_transaction_no_variants['v2ProductName'] = df_itemsets_by_transaction_no_variants.v2ProductName.apply(lambda x: list(set(x)))
m = df_itemsets_by_transaction_no_variants.v2ProductName.apply(len) > 1
df_itemsets_by_transaction_no_variants = df_itemsets_by_transaction_no_variants[m].copy()


df_itemsets_by_transaction_no_variants[df_itemsets_by_transaction_no_variants.v2ProductName.apply(lambda x: len(x) > 1)].head()


# ## Finding Association Rules

# ### Frequent itemsets (including product variants)

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


get_ipython().run_line_magic('pinfo', 'TransactionEncoder')


transactions = df_itemsets_by_transaction['fullProductName'].tolist()


te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_te = pd.DataFrame(te_ary, columns=te.columns_)


df_te.head()


frequent_itemsets = apriori(df_te, min_support=0.001, use_colnames=True)


# [source -> mlxtend docs](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/#association-rules-generation-from-frequent-itemsets)
# 
# > Typically, support is used to measure the abundance or frequency (often interpreted as significance or importance) of an itemset in a database

frequent_itemsets.sort_values('support', ascending=False).head(10)


(frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x) > 1)]
     .sort_values('support', ascending=False).head(10))


m = (
    (frequent_itemsets.itemsets.apply(len) == 2)
    & (frequent_itemsets.itemsets.apply(lambda x: all(('Sunglasses' in _x for _x in x))))
)
data_1 = frequent_itemsets[m].copy()
data_2 = frequent_itemsets[m].copy()
data_1['item_1'] = data_1.itemsets.apply(lambda x: list(x)[0])
data_1['item_2'] = data_1.itemsets.apply(lambda x: list(x)[1])
data_2['item_1'] = data_2.itemsets.apply(lambda x: list(x)[1])
data_2['item_2'] = data_2.itemsets.apply(lambda x: list(x)[0])
data = pd.concat((data_1, data_2))

data_zeros = pd.DataFrame([[0, ''] + list(x) for x in combinations(data[['item_1', 'item_2']].values.flatten(), 2)], columns=data.columns)
data = pd.concat((data, data_zeros)).groupby(['item_1', 'item_2']).support.sum().reset_index().pivot('item_1', 'item_2', 'support')

print('Support for pairs of sunglasses in transactions:')

sns.heatmap(data, cmap='YlGnBu')
plt.xlabel('')
plt.ylabel('')
savefig('association_rules_frequent_sunglasses_combinations')


# How about stuff that doesn't include sunglasses ;)

(frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x) > 1)
                    & (~(frequent_itemsets.itemsets.astype(str).str.contains('Sunglasses')))]
     .sort_values('support', ascending=False).head(10))


most_frequent_itemsets = (frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x) > 1)
                    & (~(frequent_itemsets.itemsets.astype(str).str.contains('Sunglasses')))]
                     .sort_values('support', ascending=False).head(10))

data_1 = most_frequent_itemsets.copy()
data_2 = most_frequent_itemsets.copy()
data_1['item_1'] = data_1.itemsets.apply(lambda x: list(x)[0])
data_1['item_2'] = data_1.itemsets.apply(lambda x: list(x)[1])
data_2['item_1'] = data_2.itemsets.apply(lambda x: list(x)[1])
data_2['item_2'] = data_2.itemsets.apply(lambda x: list(x)[0])
data = pd.concat((data_1, data_2))

data_zeros = pd.DataFrame([[0, ''] + list(x) for x in combinations(data[['item_1', 'item_2']].values.flatten(), 2)], columns=data.columns)
data = pd.concat((data, data_zeros)).groupby(['item_1', 'item_2']).support.sum().reset_index().pivot('item_1', 'item_2', 'support')

print('Support for top pairs of items:')

sns.heatmap(data, cmap='YlGnBu')
plt.xlabel('')
plt.ylabel('')
savefig('association_rules_frequent_item_combinations_10_including_variations')


plt.figure(figsize=(14, 10))

most_frequent_itemsets = (frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x) > 1)
                    & (~(frequent_itemsets.itemsets.astype(str).str.contains('Sunglasses')))]
                     .sort_values('support', ascending=False).head(50))

data_1 = most_frequent_itemsets.copy()
data_2 = most_frequent_itemsets.copy()
data_1['item_1'] = data_1.itemsets.apply(lambda x: list(x)[0])
data_1['item_2'] = data_1.itemsets.apply(lambda x: list(x)[1])
data_2['item_1'] = data_2.itemsets.apply(lambda x: list(x)[1])
data_2['item_2'] = data_2.itemsets.apply(lambda x: list(x)[0])
data = pd.concat((data_1, data_2))

data_zeros = pd.DataFrame([[0, ''] + list(x) for x in combinations(data[['item_1', 'item_2']].values.flatten(), 2)], columns=data.columns)
data = pd.concat((data, data_zeros)).groupby(['item_1', 'item_2']).support.sum().reset_index().pivot('item_1', 'item_2', 'support')

print('Support for top pairs of items:')

sns.heatmap(data, cmap='YlGnBu')
plt.xlabel('')
plt.ylabel('')
savefig('association_rules_frequent_item_combinations_50_including_variations')


# Notice how laptop stickers are a common add-on item for a large variety of other products.

# ### Association rules (including product variants)
# 
# Mining association rules from item sets. Note:
# 
# [source -> mlxtend docs](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/#association-rules-generation-from-frequent-itemsets)
# 
# > The confidence of a rule A->C is the probability of seeing the consequent in a transaction given that it also contains the antecedent

rules = association_rules(frequent_itemsets)


rules.sort_values('confidence', ascending=False).head()


# Loads of these have high confidence, but they are not very frequent (low support).
# 
# Given the nature of this dataset, I think it's better to sort by `support`

rules.sort_values('antecedent support', ascending=False).head(10)


m_yellow_sunglasses = df_itemsets_by_transaction.fullProductName.apply(lambda x: 'Google Sunglasses (YELLOW)' in x)
m_blue_sunglasses = df_itemsets_by_transaction.fullProductName.apply(lambda x: 'Google Sunglasses (BLUE)' in x)
m_y_and_b_sunglasses = df_itemsets_by_transaction.fullProductName.apply(lambda x: 'Google Sunglasses (YELLOW)' in x and 'Google Sunglasses (BLUE)' in x)


print('{} transactions with yellow sunglasses'.format(m_yellow_sunglasses.sum()))
print('{} transactions with blue sunglasses'.format(m_blue_sunglasses.sum()))
print('{} transactions with both'.format(m_y_and_b_sunglasses.sum()))
print('> {:.0f}% of yellow orders have blue'.format(m_y_and_b_sunglasses.sum() / m_yellow_sunglasses.sum() * 100))
print('> {:.0f}% of blue orders have yellow'.format(m_y_and_b_sunglasses.sum() / m_blue_sunglasses.sum() * 100))


# This can be used as the basis for a recommendation engine, or to inform marketing and product design decisions.
# 
# Ignoring the sunglasses...

m_no_sunglasses = (~(rules.antecedents.astype(str).str.contains('Sunglasses'))) & (~(rules.consequents.astype(str).str.contains('Sunglasses')))
rules[m_no_sunglasses].sort_values('antecedent support', ascending=False).head(10)


rules['rule'] = rules.antecedents.apply(list).astype(str) + '  ->  ' + rules.consequents.apply(list).astype(str)


s = (
    rules[m_no_sunglasses].sort_values('support', ascending=False)
         .head(10)[['rule', 'support', 'confidence']]
         .style
)
s.bar(subset=['support', 'confidence'], color='#6670ff')


# ### Frequent items & association rules (excluding product variants)

df_itemsets_by_transaction_no_variants.head()


transactions = df_itemsets_by_transaction_no_variants['v2ProductName'].tolist()


te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_te = pd.DataFrame(te_ary, columns=te.columns_)


df_te.head()


frequent_itemsets = apriori(df_te, min_support=0.001, use_colnames=True)


# [source -> mlxtend docs](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/#association-rules-generation-from-frequent-itemsets)
# 
# > Typically, support is used to measure the abundance or frequency (often interpreted as significance or importance) of an itemset in a database

frequent_itemsets.sort_values('support', ascending=False).head(10)


(frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x) > 1)]
     .sort_values('support', ascending=False).head(10))


most_frequent_itemsets = (frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x) > 1)]
                     .sort_values('support', ascending=False).head(10))

data_1 = most_frequent_itemsets.copy()
data_2 = most_frequent_itemsets.copy()
data_1['item_1'] = data_1.itemsets.apply(lambda x: list(x)[0])
data_1['item_2'] = data_1.itemsets.apply(lambda x: list(x)[1])
data_2['item_1'] = data_2.itemsets.apply(lambda x: list(x)[1])
data_2['item_2'] = data_2.itemsets.apply(lambda x: list(x)[0])
data = pd.concat((data_1, data_2))

data_zeros = pd.DataFrame([[0, ''] + list(x) for x in combinations(data[['item_1', 'item_2']].values.flatten(), 2)], columns=data.columns)
data = pd.concat((data, data_zeros)).groupby(['item_1', 'item_2']).support.sum().reset_index().pivot('item_1', 'item_2', 'support')

print('Support for top pairs of items:')

sns.heatmap(data, cmap='YlGnBu')
plt.xlabel('')
plt.ylabel('')
savefig('association_rules_frequent_item_combinations_10')


plt.figure(figsize=(14, 10))

most_frequent_itemsets = (frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x) > 1)]
                     .sort_values('support', ascending=False).head(50))

data_1 = most_frequent_itemsets.copy()
data_2 = most_frequent_itemsets.copy()
data_1['item_1'] = data_1.itemsets.apply(lambda x: list(x)[0])
data_1['item_2'] = data_1.itemsets.apply(lambda x: list(x)[1])
data_2['item_1'] = data_2.itemsets.apply(lambda x: list(x)[1])
data_2['item_2'] = data_2.itemsets.apply(lambda x: list(x)[0])
data = pd.concat((data_1, data_2))

data_zeros = pd.DataFrame([[0, ''] + list(x) for x in combinations(data[['item_1', 'item_2']].values.flatten(), 2)], columns=data.columns)
data = pd.concat((data, data_zeros)).groupby(['item_1', 'item_2']).support.sum().reset_index().pivot('item_1', 'item_2', 'support')

print('Support for top pairs of items:')

sns.heatmap(data, cmap='YlGnBu')
plt.xlabel('')
plt.ylabel('')
savefig('association_rules_frequent_item_combinations_50')


rules = association_rules(frequent_itemsets)


rules.sort_values('confidence', ascending=False).head()


rules.sort_values('antecedent support', ascending=False).head(10)


rules['rule'] = rules.antecedents.apply(list).astype(str) + '  ->  ' + rules.consequents.apply(list).astype(str)


s = (
    rules.sort_values('support', ascending=False)
         .head(10)[['rule', 'support', 'confidence']]
         .style
)
s.bar(subset=['support', 'confidence'], color='#6670ff')


from IPython.display import HTML
HTML('<style>div.text_cell_render{font-size:130%;padding-top:50px;padding-bottom:50px}</style>')

