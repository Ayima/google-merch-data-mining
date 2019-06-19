
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

def savefig(name):
    plt.savefig(f'../../figures/{name}.png', bbox_inches='tight', dpi=300)

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
    
get_ipython().run_line_magic('reload_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'pandas, numpy')


from google.cloud import bigquery
from slugify import slugify


from dotenv import load_dotenv
load_dotenv('../../.env')


client = bigquery.Client()


# # Sales Forecasting
# Alex's development notebook for sales forecasting.

# ## Read data

# ### Read from bigquery

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


# ### Read `jsonl` from google drive

get_ipython().run_cell_magic('time', '', '"""\nUsing gdrive jsonl\n"""\nERRORS = []\n\ndef pull_daily_data(verbose=False, raise_errors=False):\n    dataset = sorted(glob.glob(\'/Volumes/GoogleDrive/My Drive/bigquery_ga_sample/*.jsonl\'))\n\n    data = []\n    for table in tqdm_notebook(dataset):\n        if verbose:\n            print(\'Scanning {}\'.format(table))\n        with open(table, \'r\') as f:\n            table_data = []\n            for line in f:\n                d = json.loads(line)\n                date = d[\'date\']\n                d = d[\'totals\']\n                try:\n                    table_data.append([\n                        date,\n                        d[\'visits\'],\n                        d[\'pageviews\'],\n                        d[\'transactions\'],\n                        d[\'transactionRevenue\'],\n                    ])\n                except Exception as e:\n                    if verbose:\n                        print(\'Error raised when reading row:\\n{}\'.format(e))\n                    ERRORS.append([table, e])\n                    if raise_errors:\n                        raise(e)\n\n            cols = [\'date\', \'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']\n            results = (\n                pd.DataFrame(table_data, columns=cols)\n                    .groupby(\'date\')[[\'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']]\n                    .sum().reset_index()\n            )\n            data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df\n\njsonl_gdrive_results = pull_daily_data()')


jsonl_gdrive_results


# ### Read `jsonl` from local

get_ipython().run_cell_magic('time', '', '"""\nUsing local jsonl\n"""\nERRORS = []\n\ndef pull_daily_data(verbose=False, raise_errors=False):\n    dataset = sorted(glob.glob(\'../../data/raw/*.jsonl\'))\n\n    data = []\n    for table in tqdm_notebook(dataset):\n        if verbose:\n            print(\'Scanning {}\'.format(table))\n        with open(table, \'r\') as f:\n            table_data = []\n            for line in f:\n                d = json.loads(line)\n                date = d[\'date\']\n                d = d[\'totals\']\n                try:\n                    table_data.append([\n                        date,\n                        d[\'visits\'],\n                        d[\'pageviews\'],\n                        d[\'transactions\'],\n                        d[\'transactionRevenue\'],\n                    ])\n                except Exception as e:\n                    if verbose:\n                        print(\'Error raised when reading row:\\n{}\'.format(e))\n                    ERRORS.append([table, e])\n                    if raise_errors:\n                        raise(e)\n\n            cols = [\'date\', \'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']\n            results = (\n                pd.DataFrame(table_data, columns=cols)\n                    .groupby(\'date\')[[\'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']]\n                    .sum().reset_index()\n            )\n            data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df\n\njsonl_results = pull_daily_data()')


df = jsonl_results.copy()
df.date = pd.to_datetime(df.date)
df.to_csv('../../data/interim/sales_forecast_raw.csv', index=False)


def load_file(f_path):
    if not os.path.exists(f_path):
        print('No data found. Run data load script above.')
        return
    print('Loading {}'.format(f_path))
    df = pd.read_csv(f_path)
    df.date = pd.to_datetime(df.date)
    return df

# Looking forward to walrus operator for stuff like this...
tmp = load_file('../../data/interim/sales_forecast_raw.csv')
if tmp is not None:
    print('Loading from file')
    df = tmp.copy()
    del tmp


df.head()


df['week'] = df.date.apply(lambda x: x.strftime('%W'))
df['year'] = df.date.apply(lambda x: x.strftime('%Y'))
df['week_start'] = df[['week', 'year']].apply(
    lambda x: datetime.datetime.strptime('{}-{}-1'.format(x.year, x.week), '%Y-%W-%w'),
    axis=1
)


df.dtypes


# How does the data look? Is there seasonality that I can predict?

df.visits.plot()


df.groupby('week_start').visits.sum().plot()


df.transactions.plot()


df.groupby('week_start').transactions.sum().plot()


# We should throw out the first and last week (for these charts)

df_ = df[(df.week_start > df.week_start.min()) & (df.week_start < df.week_start.max())].copy()


df_.groupby('week_start').visits.sum().plot()


df_.groupby('week_start').transactions.sum().plot()


# That's better!

# ## Forcasting with Facebook Prophet

from fbprophet import Prophet


# ### Transactions

df_prophet = df[['date', 'transactions']]    .rename(columns={'date': 'ds', 'transactions': 'y'})
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet()
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)
fig1 = m.plot(forecast)


fig2 = m.plot_components(forecast)


# Trying to fit the data better

get_ipython().run_line_magic('pinfo', 'Prophet')


# Weekly seasonality

df_prophet = df[['date', 'transactions']]    .rename(columns={'date': 'ds', 'transactions': 'y'})
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet(weekly_seasonality=True)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)
fig1 = m.plot(forecast)


# Tuning `changepoint_prior_scale`

df_prophet = df[['date', 'transactions']]    .rename(columns={'date': 'ds', 'transactions': 'y'})
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet(weekly_seasonality=True, changepoint_prior_scale=1)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)
fig1 = m.plot(forecast)


df_prophet = df[['date', 'transactions']]    .rename(columns={'date': 'ds', 'transactions': 'y'})
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet(weekly_seasonality=True, changepoint_prior_scale=.1)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)
fig1 = m.plot(forecast)


df_prophet = df[['date', 'transactions']]    .rename(columns={'date': 'ds', 'transactions': 'y'})
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet(weekly_seasonality=True, changepoint_prior_scale=.5)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)
fig1 = m.plot(forecast)

plt.ylim(0, 100)


# Trying out `daily_seasonality`

df_prophet = df[['date', 'transactions']]    .rename(columns={'date': 'ds', 'transactions': 'y'})
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet(daily_seasonality=True, weekly_seasonality=True)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)
fig1 = m.plot(forecast)


# This looks better! The trick is to add `yearly_seasonality`

df_prophet = df[['date', 'transactions']]    .rename(columns={'date': 'ds', 'transactions': 'y'})
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)
fig1 = m.plot(forecast)


fig2 = m.plot_components(forecast)


get_ipython().run_line_magic('pinfo', 'Prophet')


# `n_changepoints`

df_prophet = df[['date', 'transactions']]    .rename(columns={'date': 'ds', 'transactions': 'y'})
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet(weekly_seasonality=True, yearly_seasonality=True, n_changepoints=2)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)
fig1 = m.plot(forecast)


df_prophet = df[['date', 'transactions']]    .rename(columns={'date': 'ds', 'transactions': 'y'})
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet(weekly_seasonality=True, yearly_seasonality=True, n_changepoints=0)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)
fig1 = m.plot(forecast)


# `seasonality_prior_scale`

df_prophet = df[['date', 'transactions']]    .rename(columns={'date': 'ds', 'transactions': 'y'})
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet(weekly_seasonality=True, yearly_seasonality=True, seasonality_prior_scale=0.01)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)
fig1 = m.plot(forecast)


df_prophet = df[['date', 'transactions']]    .rename(columns={'date': 'ds', 'transactions': 'y'})
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False, seasonality_prior_scale=0.1)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)
fig1 = m.plot(forecast)


# ### Sales

df_prophet = df[['date', 'transactionRevenue']]    .rename(columns={'date': 'ds', 'transactionRevenue': 'y'})
df_prophet['y'] = df_prophet['y'] / 1e6
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet(weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            seasonality_prior_scale=0.1)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)
fig1 = m.plot(forecast)


# Here we give more weight to the yearly seasonality prior, but it predicts too aggrestive growth for me

df_prophet = df[['date', 'transactionRevenue']]    .rename(columns={'date': 'ds', 'transactionRevenue': 'y'})
df_prophet['y'] = df_prophet['y'] / 1e6
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet(weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            seasonality_prior_scale=1)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)
fig1 = m.plot(forecast)


# ### Predictions by quarter

df_prophet = df[['date', 'transactions']]    .rename(columns={'date': 'ds', 'transactions': 'y'})
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False, seasonality_prior_scale=0.1)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)


forecast['date'] = pd.to_datetime(forecast['ds'])


def add_quarters(df):
    df['quarter'] = float('nan')
    df['quarter_num'] = float('nan')
    
    df.loc[(df.date >= datetime.datetime(2016, 7, 1))&(df.date < datetime.datetime(2016, 10, 1)), 'quarter'] = '2016 Q3'
    df.loc[(df.date >= datetime.datetime(2016, 10, 1))&(df.date < datetime.datetime(2017, 1, 1)), 'quarter'] = '2016 Q4'
    df.loc[(df.date >= datetime.datetime(2017, 1, 1))&(df.date < datetime.datetime(2017, 4, 1)), 'quarter'] = '2017 Q1'
    df.loc[(df.date >= datetime.datetime(2017, 4, 1))&(df.date < datetime.datetime(2017, 7, 1)), 'quarter'] = '2017 Q2'
    df.loc[(df.date >= datetime.datetime(2017, 7, 1))&(df.date < datetime.datetime(2017, 10, 1)), 'quarter'] = '2017 Q3'
    df.loc[(df.date >= datetime.datetime(2017, 10, 1))&(df.date < datetime.datetime(2018, 1, 1)), 'quarter'] = '2017 Q4'
    df.loc[(df.date >= datetime.datetime(2018, 1, 1))&(df.date < datetime.datetime(2018, 4, 1)), 'quarter'] = '2018 Q1'
    df.loc[(df.date >= datetime.datetime(2018, 4, 1))&(df.date < datetime.datetime(2018, 7, 1)), 'quarter'] = '2018 Q2'
    df.loc[(df.date >= datetime.datetime(2018, 7, 1))&(df.date < datetime.datetime(2018, 10, 1)), 'quarter'] = '2018 Q3'
    df.loc[(df.date >= datetime.datetime(2018, 10, 1))&(df.date < datetime.datetime(2019, 1, 1)), 'quarter'] = '2018 Q4'

    df.loc[(df.date >= datetime.datetime(2016, 7, 1))&(df.date < datetime.datetime(2016, 10, 1)), 'quarter_num'] = 1
    df.loc[(df.date >= datetime.datetime(2016, 10, 1))&(df.date < datetime.datetime(2017, 1, 1)), 'quarter_num'] = 2
    df.loc[(df.date >= datetime.datetime(2017, 1, 1))&(df.date < datetime.datetime(2017, 4, 1)), 'quarter_num'] = 3
    df.loc[(df.date >= datetime.datetime(2017, 4, 1))&(df.date < datetime.datetime(2017, 7, 1)), 'quarter_num'] = 4
    df.loc[(df.date >= datetime.datetime(2017, 7, 1))&(df.date < datetime.datetime(2017, 10, 1)), 'quarter_num'] = 5
    df.loc[(df.date >= datetime.datetime(2017, 10, 1))&(df.date < datetime.datetime(2018, 1, 1)), 'quarter_num'] = 6
    df.loc[(df.date >= datetime.datetime(2018, 1, 1))&(df.date < datetime.datetime(2018, 4, 1)), 'quarter_num'] = 7
    df.loc[(df.date >= datetime.datetime(2018, 4, 1))&(df.date < datetime.datetime(2018, 7, 1)), 'quarter_num'] = 8
    df.loc[(df.date >= datetime.datetime(2018, 7, 1))&(df.date < datetime.datetime(2018, 10, 1)), 'quarter_num'] = 9
    df.loc[(df.date >= datetime.datetime(2018, 10, 1))&(df.date < datetime.datetime(2019, 1, 1)), 'quarter_num'] = 10
    return df


forecast = add_quarters(forecast)


forecast.quarter.value_counts()


# We only have partial data for 2016 & 2018 Q3, so we'll want to filter these out

m = (forecast.quarter != '2016 Q3') & (forecast.quarter != '2018 Q3')
s_transactions = (
    forecast[m].groupby(['quarter_num', 'quarter'])
        .yhat.sum().reset_index()
        .set_index('quarter_num').sort_index(ascending=True)
        .set_index('quarter')['yhat']
)
s_transactions.apply(lambda x: '{:,}'.format(round(x)))


df_prophet = df[['date', 'transactionRevenue']]    .rename(columns={'date': 'ds', 'transactionRevenue': 'y'})
df_prophet['y'] = df_prophet['y'] / 1e6
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet(weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            seasonality_prior_scale=0.1)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)

forecast['date'] = pd.to_datetime(forecast['ds'])
forecast = add_quarters(forecast)

m = (forecast.quarter != '2016 Q3') & (forecast.quarter != '2018 Q3')
s_transactionRevenue = (
    forecast[m].groupby(['quarter_num', 'quarter'])
        .yhat.sum().reset_index()
        .set_index('quarter_num').sort_index(ascending=True)
        .set_index('quarter')['yhat']
)
s_transactionRevenue.apply(lambda x: '${:,}'.format(round(x)))


# Display these results in a dataframe

# Get the actual sales (as opposed to predicted above)

df = add_quarters(df)
m = (df.quarter != '2016 Q3') & (df.quarter != '2018 Q3')
s_actual_transactionRevenue = (
    df[m].groupby(['quarter_num', 'quarter'])
        .transactionRevenue.sum().reset_index()
        .set_index('quarter_num').sort_index(ascending=True)
        .set_index('quarter')['transactionRevenue'] / 1e6
)
m.apply(lambda x: '${:,}'.format(round(x)))


forecast_results = pd.DataFrame({
    'Reporting Period': ['Q4', 'Q1', 'Q2'],
    'Prev Year': [386901, 337599, 402070],
    'Forecasted': [509898, 454149, 515331],
})
forecast_results['YoY (%)'] = ((forecast_results['Forecasted'] - forecast_results['Prev Year'])
                                / forecast_results['Prev Year'] * 100).apply(lambda x: '{:+.0f}%'.format(x))
forecast_results['Prev Year'] = forecast_results['Prev Year'].apply(lambda x: '{:,}'.format(x))
forecast_results['Forecasted'] = forecast_results['Forecasted'].apply(lambda x: '{:,}'.format(x))
forecast_results.set_index('Reporting Period', inplace=True)
forecast_results.to_csv('../../data/interim/sales_forecast.csv')


forecast_results


# ## Forecasting by product

# Looking a couple samples...

def search():
    data = []
    i = 0
    for f_name in glob.glob('../../data/raw/*.jsonl'):
        print('scanning {}'.format(f_name))
        with open(f_name, 'r') as f:
            for line in tqdm_notebook(f.readlines()):
                i += 1
                d = json.loads(line.strip())
                if d['totals']['transactions']:
                    data.append(d)
                    return data

print(json.dumps(search()))

# with open('/tmp/temp.json', 'w') as f:
#     json.dump(data, f)


# ### Read from `jsonl` local

from typing import List, Tuple, Dict


get_ipython().run_cell_magic('time', '', '"""\nUsing local jsonl\n"""\nERRORS = []\n\ndef pull_daily_product_sales(\n    verbose=False,\n    raise_errors=False,\n    test=False,\n) -> Tuple[pd.DataFrame, dict]:\n    dataset = sorted(glob.glob(\'../../data/raw/*.jsonl\'))\n\n    data = []\n    for table in tqdm_notebook(dataset):\n        if verbose:\n            print(\'Scanning {}\'.format(table))\n        with open(table, \'r\') as f:\n            for line in f:\n                d = json.loads(line)\n                try:\n                    if not d[\'totals\'][\'transactions\']:\n                        # No purchases, continue to next visitor\n                        continue\n                    for hit in d[\'hits\']:\n                        for product in hit[\'product\']:\n                            if product[\'productRevenue\']:\n                                data.append({\n                                    \'date\': d[\'date\'],\n                                    \'visitId\': d[\'visitId\'],\n                                    \'fullVisitorId\': d[\'fullVisitorId\'],\n                                    \'product\': product,\n                                })\n                except Exception as e:\n                    if verbose:\n                        print(\'Error raised when reading row:\\n{}\'.format(e))\n                    ERRORS.append([table, e])\n                    if raise_errors:\n                        raise(e)\n                        \n        if test and (table == dataset[1]):\n            break\n\n    cols_main = [\'date\', \'visitId\', \'fullVisitorId\']\n    cols_product = [\n        \'productSKU\', \'v2ProductName\', \'v2ProductCategory\', \'productVariant\',\n        \'productRevenue\', \'productQuantity\', \'productRefundAmount\'\n    ]\n    df_data = [\n        [d.get(col, float(\'nan\')) for col in cols_main]\n        + [d[\'product\'].get(col) for col in cols_product]\n        for d in data\n    ]\n    df = pd.DataFrame(df_data, columns=(cols_main+cols_product))\n    return df, data\n\njsonl_product_results, nosql_data = pull_daily_product_sales(raise_errors=True, test=True)')


jsonl_product_results, nosql_data = pull_daily_product_sales()


df = jsonl_product_results.copy()
df.date = pd.to_datetime(df.date)
df.to_csv('../../data/interim/product_sales_forecast_raw.csv', index=False)


def load_file(f_path):
    if not os.path.exists(f_path):
        print('No data found. Run data load script above.')
        return
    print('Loading {}'.format(f_path))
    df = pd.read_csv(f_path)
    df.date = pd.to_datetime(df.date)
    return df

tmp = load_file('../../data/interim/product_sales_forecast_raw.csv')
if tmp is not None:
    print('Loading from file')
    df = tmp.copy()
    del tmp


df.head()


# ### Read from bigquery

get_ipython().run_cell_magic('time', '', '"""\nUsing bigquery\n"""\n\ndef pull_daily_product_sales(verbose=False):\n    dataset = client.get_dataset(\'bigquery-public-data.google_analytics_sample\')\n\n    data = []\n    for table in tqdm_notebook(list(client.list_tables(dataset))):\n        if verbose:\n            print(\'Querying {}\'.format(table.table_id))\n        query_job = client.query(\'\'\'\n        SELECT\n            h.item.productName AS other_purchased_products,\n            COUNT(h.item.productName) AS quantity\n        FROM `bigquery-public-data.google_analytics_sample.{}`,\n            UNNEST(hits) as h\n        WHERE (\n            fullVisitorId IN (\n                SELECT fullVisitorId\n                FROM `bigquery-public-data.google_analytics_sample.{}`,\n                    UNNEST(hits) as h\n                WHERE h.item.productName CONTAINS \'Product Item Name A\'\n                AND totals.transactions>=1\n                GROUP BY fullVisitorId\n            )\n            AND h.item.productName IS NOT NULL\n            AND h.item.productName != \'Product Item Name A\'\n        )\n        GROUP BY other_purchased_products\n        ORDER BY quantity DESC;\n        \'\'\'.format(table.table_id, table.table_id))\n        results = query_job.result().to_dataframe()\n        results.columns = [\'date\', \'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']\n        data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df\n\nbq_results = pull_daily_product_sales()')


bq_results.head()


# ### Exploration

# What's going on with refunds?

(~(df.productRefundAmount.isnull())).sum()


# They are all null. That is good.

# How many _different_ products are purchased together?

fig = plt.figure(figsize=(8, 8))
s = df.groupby('visitId').size().value_counts().sort_index(ascending=False)
s.plot.barh(color='b')


# How are product quantities distributed?

fig = plt.figure(figsize=(8, 8))
s = df.productQuantity.value_counts().sort_index(ascending=False).tail(20)
s.plot.barh(color='b')


# Top selling products

df.v2ProductName.value_counts(ascending=False).head(10)


# Top grossing products

(df.groupby('v2ProductName').productRevenue.sum() / 1e6).sort_values(ascending=False).head(10)


# ### Forecasts by product name

# Daily forcasts are choppy... 

def product_forcast(df, product_name) -> pd.DataFrame:

    m = df.v2ProductName == product_name
    print('Found {} product transactions'.format(m.sum()))
    if m.sum() == 0:
        print('Returning empty DataFrame')
        return pd.DataFrame

    df_prophet = df[m][['date', 'productRevenue']]        .rename(columns={'date': 'ds', 'productRevenue': 'y'})
    df_prophet['y'] = df_prophet['y'] / 1e6
    df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

    m = Prophet(weekly_seasonality=True,
                yearly_seasonality=True,
                daily_seasonality=False,
                seasonality_prior_scale=0.01)
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=365, freq='D')
    forecast = m.predict(future)
    fig = m.plot(forecast)
    
    return forecast, fig


forecast, fig = product_forcast(df, 'Google Men\'s  Zip Hoodie')

plt.ylim(0, 100)
plt.show()


# Weekly forcasts

def product_forcast(df, product_name) -> pd.DataFrame:

    m = df.v2ProductName == product_name
    print('Found {} product transactions'.format(m.sum()))
    if m.sum() == 0:
        print('Returning empty DataFrame')
        return pd.DataFrame

    df_prophet = df[m][['date', 'productRevenue']]        .rename(columns={'date': 'ds', 'productRevenue': 'y'})
    df_prophet['y'] = df_prophet['y'] / 1e6
    df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

    m = Prophet(weekly_seasonality=True,
                yearly_seasonality=True,
                daily_seasonality=False,
                seasonality_prior_scale=0.1)
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=52, freq='7D')
    forecast = m.predict(future)
    fig = m.plot(forecast)
    
    return forecast, fig


forecast, fig = product_forcast(df, 'Google Men\'s  Zip Hoodie')

plt.show()


# Trying `seasonality_mode='multiplicative'`

def product_forcast(df, product_name) -> pd.DataFrame:

    m = df.v2ProductName == product_name
    print('Found {} product transactions'.format(m.sum()))
    if m.sum() == 0:
        print('Returning empty DataFrame')
        return pd.DataFrame

    df_prophet = df[m][['date', 'productRevenue']]        .rename(columns={'date': 'ds', 'productRevenue': 'y'})
    df_prophet['y'] = df_prophet['y'] / 1e6
    df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

    m = Prophet(seasonality_mode='multiplicative')
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=52, freq='7D')
    forecast = m.predict(future)
    fig = m.plot(forecast)
    
    return forecast, fig


forecast, fig = product_forcast(df, 'Google Men\'s  Zip Hoodie')

plt.show()


# No arguments...

def product_forcast(df, product_name) -> pd.DataFrame:

    m = df.v2ProductName == product_name
    print('Found {} product transactions'.format(m.sum()))
    if m.sum() == 0:
        print('Returning empty DataFrame')
        return pd.DataFrame

    df_prophet = df[m][['date', 'productRevenue']]        .rename(columns={'date': 'ds', 'productRevenue': 'y'})
    df_prophet['y'] = df_prophet['y'] / 1e6
    df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

    m = Prophet()
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=365, freq='D')
    forecast = m.predict(future)
    fig = m.plot(forecast)
    
    return forecast, fig


forecast, fig = product_forcast(df, 'Google Men\'s  Zip Hoodie')

plt.show()


# Google sunglasses

forecast, fig = product_forcast(df, 'Google Sunglasses')

plt.show()


def product_forcast(df, product_name) -> pd.DataFrame:

    m = df.v2ProductName == product_name
    print('Found {} product transactions'.format(m.sum()))
    if m.sum() == 0:
        print('Returning empty DataFrame')
        return pd.DataFrame

    df_prophet = df[m][['date', 'productRevenue']]        .rename(columns={'date': 'ds', 'productRevenue': 'y'})
    df_prophet['y'] = df_prophet['y'] / 1e6
    df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

    m = Prophet(seasonality_mode='multiplicative')
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=365, freq='D')
    forecast = m.predict(future)
    fig = m.plot(forecast)
    
    return forecast, fig


forecast, fig = product_forcast(df, 'Google Sunglasses')

plt.show()


def product_forcast(df, product_name) -> pd.DataFrame:

    m = df.v2ProductName == product_name
    print('Found {} product transactions'.format(m.sum()))
    if m.sum() == 0:
        print('Returning empty DataFrame')
        return pd.DataFrame

    df_prophet = df[m][['date', 'productRevenue']]        .rename(columns={'date': 'ds', 'productRevenue': 'y'})
    df_prophet['y'] = df_prophet['y'] / 1e6
    df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=365, freq='D')
    forecast = m.predict(future)
    fig = m.plot(forecast)
    
    return forecast, fig


forecast, fig = product_forcast(df, 'Google Sunglasses')
plt.show()


forecast, fig = product_forcast(df, 'Google Men\'s  Zip Hoodie')
plt.show()


# Clearly there was some crazy stuff going on with the hoodies for a few days. It may not be possilbe to predict this kind of event.
# 
# I am going to remove these outliers (by computing standard deviation and filtering out e.g. `>3s.d.`). Then I'll take that missing revenue and add it back into the quarterly predictions, splitting evenly.

def product_forcast(
    df,
    product_name,
    ignore_std=0,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
) -> pd.DataFrame:

    m = df.v2ProductName == product_name
    if ignore_std:
        len_0 = m.sum()
        m_outliers = df.productRevenue < (df.loc[m, 'productRevenue'].mean() + df.loc[m, 'productRevenue'].std()*ignore_std)
        print('Ignoring {} outlier points'.format(len_0 - m.sum()))
        
    print('Found {} product transactions'.format(m.sum()))
    if m.sum() == 0:
        print('Returning empty DataFrame')
        return pd.DataFrame

    df_prophet = df[m][['date', 'productRevenue']]        .rename(columns={'date': 'ds', 'productRevenue': 'y'})
    df_prophet['y'] = df_prophet['y'] / 1e6
    df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

    m = Prophet(yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality)
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=365, freq='D')
    forecast = m.predict(future)
    fig = m.plot(forecast)
    
    return forecast, fig


forecast, fig = product_forcast(df, 'Google Sunglasses', ignore_std=1)
plt.show()


forecast, fig = product_forcast(df, 'Google Men\'s  Zip Hoodie', ignore_std=1)
plt.show()


# Weekly seasonality

forecast, fig = product_forcast(df, 'Google Men\'s  Zip Hoodie', ignore_std=1, weekly_seasonality=False)
plt.show()


# Making forecasts by quarter

def add_quarters(df):
    df['quarter'] = float('nan')
    df['quarter_num'] = float('nan')
    
    df.loc[(df.date >= datetime.datetime(2016, 7, 1))&(df.date < datetime.datetime(2016, 10, 1)), 'quarter'] = '2016 Q3'
    df.loc[(df.date >= datetime.datetime(2016, 10, 1))&(df.date < datetime.datetime(2017, 1, 1)), 'quarter'] = '2016 Q4'
    df.loc[(df.date >= datetime.datetime(2017, 1, 1))&(df.date < datetime.datetime(2017, 4, 1)), 'quarter'] = '2017 Q1'
    df.loc[(df.date >= datetime.datetime(2017, 4, 1))&(df.date < datetime.datetime(2017, 7, 1)), 'quarter'] = '2017 Q2'
    df.loc[(df.date >= datetime.datetime(2017, 7, 1))&(df.date < datetime.datetime(2017, 10, 1)), 'quarter'] = '2017 Q3'
    df.loc[(df.date >= datetime.datetime(2017, 10, 1))&(df.date < datetime.datetime(2018, 1, 1)), 'quarter'] = '2017 Q4'
    df.loc[(df.date >= datetime.datetime(2018, 1, 1))&(df.date < datetime.datetime(2018, 4, 1)), 'quarter'] = '2018 Q1'
    df.loc[(df.date >= datetime.datetime(2018, 4, 1))&(df.date < datetime.datetime(2018, 7, 1)), 'quarter'] = '2018 Q2'
    df.loc[(df.date >= datetime.datetime(2018, 7, 1))&(df.date < datetime.datetime(2018, 10, 1)), 'quarter'] = '2018 Q3'
    df.loc[(df.date >= datetime.datetime(2018, 10, 1))&(df.date < datetime.datetime(2019, 1, 1)), 'quarter'] = '2018 Q4'

    df.loc[(df.date >= datetime.datetime(2016, 7, 1))&(df.date < datetime.datetime(2016, 10, 1)), 'quarter_num'] = 1
    df.loc[(df.date >= datetime.datetime(2016, 10, 1))&(df.date < datetime.datetime(2017, 1, 1)), 'quarter_num'] = 2
    df.loc[(df.date >= datetime.datetime(2017, 1, 1))&(df.date < datetime.datetime(2017, 4, 1)), 'quarter_num'] = 3
    df.loc[(df.date >= datetime.datetime(2017, 4, 1))&(df.date < datetime.datetime(2017, 7, 1)), 'quarter_num'] = 4
    df.loc[(df.date >= datetime.datetime(2017, 7, 1))&(df.date < datetime.datetime(2017, 10, 1)), 'quarter_num'] = 5
    df.loc[(df.date >= datetime.datetime(2017, 10, 1))&(df.date < datetime.datetime(2018, 1, 1)), 'quarter_num'] = 6
    df.loc[(df.date >= datetime.datetime(2018, 1, 1))&(df.date < datetime.datetime(2018, 4, 1)), 'quarter_num'] = 7
    df.loc[(df.date >= datetime.datetime(2018, 4, 1))&(df.date < datetime.datetime(2018, 7, 1)), 'quarter_num'] = 8
    df.loc[(df.date >= datetime.datetime(2018, 7, 1))&(df.date < datetime.datetime(2018, 10, 1)), 'quarter_num'] = 9
    df.loc[(df.date >= datetime.datetime(2018, 10, 1))&(df.date < datetime.datetime(2019, 1, 1)), 'quarter_num'] = 10
    return df


def product_forcast(
    df,
    product_name,
    ignore_std=0,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    add_back_outlier_revenue=False,
) -> pd.DataFrame:

    df_ = df[df.v2ProductName == product_name].groupby('date').productRevenue.sum().reset_index()
    m = pd.Series(True, index=df_.index)
    if ignore_std:
        m = df_.productRevenue < (df_.productRevenue.mean() + df_.productRevenue.std()*ignore_std)
        if add_back_outlier_revenue:
            outlier_sum = df_.loc[~m, 'productRevenue'].sum() / 1e6
        else:
            outlier_sum = 0
        print('Ignoring {} outlier points'.format((~m).sum()))
        
    print('Found {} product transactions'.format(m.sum()))
    if m.sum() == 0:
        print('Returning empty DataFrame')
        return pd.DataFrame()

    df_prophet = df_[m][['date', 'productRevenue']]        .rename(columns={'date': 'ds', 'productRevenue': 'y'})
    df_prophet['y'] = df_prophet['y'] / 1e6
    df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

    model = Prophet(yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=365, freq='D')
    forecast = model.predict(future)
    model.plot(forecast)
    
    print('Generating quarterly forecasts')
    forecast['date'] = pd.to_datetime(forecast['ds'])
    forecast = add_quarters(forecast)

    m_q = (forecast.quarter != '2016 Q3') & (forecast.quarter != '2018 Q3')
    s_transactions = (
        forecast[m_q].groupby(['quarter_num', 'quarter'])
            .yhat.sum().reset_index()
            .set_index('quarter_num').sort_index(ascending=True)
            .set_index('quarter')['yhat']
    )

    # Get the actual sales (as opposed to predicted above)
    df_ = add_quarters(df_)
    m_q = (df_.quarter != '2016 Q3') & (df_.quarter != '2018 Q3')
    s_actual_productRevenue = (
        df_[m & m_q].groupby(['quarter_num', 'quarter'])
            .productRevenue.sum().reset_index()
            .set_index('quarter_num').sort_index(ascending=True)
            .set_index('quarter')['productRevenue'] / 1e6
    )

    forecast_results = pd.DataFrame({
        'Reporting Period': ['Q4', 'Q1', 'Q2'],
        'Prev Year': [
            s_actual_productRevenue[s_actual_productRevenue.index=='2016 Q4'].values[0] if (s_actual_productRevenue.index=='2016 Q4').sum() else 0,
            s_actual_productRevenue[s_actual_productRevenue.index=='2017 Q1'].values[0] if (s_actual_productRevenue.index=='2017 Q1').sum() else 0,
            s_actual_productRevenue[s_actual_productRevenue.index=='2017 Q2'].values[0] if (s_actual_productRevenue.index=='2017 Q2').sum() else 0,
        ],
        'Forecasted': [
            s_transactions[s_transactions.index=='2017 Q4'].values[0] + outlier_sum/4,
            s_transactions[s_transactions.index=='2018 Q1'].values[0] + outlier_sum/4,
            s_transactions[s_transactions.index=='2018 Q2'].values[0] + outlier_sum/4,
        ],
    })
    forecast_results['YoY (%)'] = ((forecast_results['Forecasted'] - forecast_results['Prev Year'])
                                    / forecast_results['Prev Year'] * 100).apply(lambda x: '{:+.0f}%'.format(x))
    forecast_results['Prev Year'] = forecast_results['Prev Year'].apply(lambda x: '${:,.0f}'.format(x))
    forecast_results['Forecasted'] = forecast_results['Forecasted'].apply(lambda x: '${:,.0f}'.format(x))
    forecast_results.set_index('Reporting Period', inplace=True)
    forecast_results.to_csv('../../data/interim/sales_forecast_{}.csv'.format(slugify(product_name)))
    
    return forecast, forecast_results, fig


forecast, forecast_results, fig = product_forcast(
    df,
    'Google Men\'s  Zip Hoodie',
    ignore_std=1,
    weekly_seasonality=False
)
plt.show()
forecast_results


# Run for each of the top 30 Top selling products

for product in df.v2ProductName.value_counts(ascending=False).head(30).index.tolist():
    print('-'*20)
    print(Fore.RED + product + Style.RESET_ALL)
    forecast, forecast_results, fig = product_forcast(
        df,
        product,
        ignore_std=1,
        weekly_seasonality=False
    )
    plt.show()
    display(forecast_results)


from IPython.display import HTML
HTML('<style>div.text_cell_render{font-size:130%;padding-top:50px;padding-bottom:50px}</style>')

