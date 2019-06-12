
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
    f_name = '../../results/figures/{}.png'.format(name)
    plt.savefig(f_name, bbox_inches='tight', dpi=300)
    print('Saving figure to {}'.format(f_name))

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
    
get_ipython().run_line_magic('reload_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'pandas, numpy')


from slugify import slugify


from dotenv import load_dotenv
load_dotenv('../../.env')


# # Sales Forecasting
# Alex's development notebook for sales forecasting.

# ## Read data

# ### Read from BigQuery
# 
# Run this cell to read from BigQuery.

get_ipython().run_cell_magic('time', '', '"""\nUsing bigquery\n"""\n\nfrom google.cloud import bigquery\n# Using environment variables to authenticate\nclient = bigquery.Client()\n\ndef pull_daily_data(verbose=False):\n    dataset = client.get_dataset(\'bigquery-public-data.google_analytics_sample\')\n\n    data = []\n    for table in tqdm_notebook(list(client.list_tables(dataset))):\n        if verbose:\n            print(\'Querying {}\'.format(table.table_id))\n        query_job = client.query(\'\'\'\n            select\n              date,\n              sum(totals.visits),\n              sum(totals.pageviews),\n              sum(totals.transactions),\n              sum(totals.transactionRevenue)\n            from `bigquery-public-data.google_analytics_sample.{}`\n            group by date;\n        \'\'\'.format(table.table_id))\n        results = query_job.result().to_dataframe()\n        results.columns = [\'date\', \'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']\n        data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df\n\nbq_results = pull_daily_data()')


bq_results.head()


# ### Read `jsonl` from local
# 
# Run this cell if you have exported the data from BigQuery into JSON line (JSONL) files.

get_ipython().run_cell_magic('time', '', '"""\nUsing local jsonl\n"""\nERRORS = []\n\ndef pull_daily_data(verbose=False, raise_errors=False):\n    if os.getenv(\'DATA_PATH\') is None:\n        raise ValueError(\'Please set environment variable DATA_PATH\')\n\n    dataset = sorted(glob.glob(os.path.join(os.getenv(\'DATA_PATH\'), \'raw\', \'*.jsonl\')))\n    print(\'Loading from {}, etc...\'.format(\', \'.join(dataset[:3])))\n\n    data = []\n    for table in tqdm_notebook(dataset):\n        if verbose:\n            print(\'Scanning {}\'.format(table))\n        with open(table, \'r\') as f:\n            table_data = []\n            for line in f:\n                d = json.loads(line)\n                date = d[\'date\']\n                d = d[\'totals\']\n                try:\n                    table_data.append([\n                        date,\n                        d[\'visits\'],\n                        d[\'pageviews\'],\n                        d[\'transactions\'],\n                        d[\'transactionRevenue\'],\n                    ])\n                except Exception as e:\n                    if verbose:\n                        print(\'Error raised when reading row:\\n{}\'.format(e))\n                    ERRORS.append([table, e])\n                    if raise_errors:\n                        raise(e)\n\n            cols = [\'date\', \'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']\n            results = (\n                pd.DataFrame(table_data, columns=cols)\n                    .groupby(\'date\')[[\'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']]\n                    .sum().reset_index()\n            )\n            data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df\n\njsonl_results = pull_daily_data()')


df = jsonl_results.copy()
df.date = pd.to_datetime(df.date)

f_path = os.path.join(os.getenv('DATA_PATH'), 'interim', 'sales_forecast_raw.csv')
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

df = load_file('../../data/interim/sales_forecast_raw.csv')


df.head()


df['week'] = df.date.apply(lambda x: x.strftime('%W'))
df['year'] = df.date.apply(lambda x: x.strftime('%Y'))
df['week_start'] = df[['week', 'year']].apply(
    lambda x: datetime.datetime.strptime('{}-{}-1'.format(x.year, x.week), '%Y-%W-%w'),
    axis=1
)


df.dtypes


# How does the data look? Is there seasonality that I can predict?
# 
# Note: we'll throw out the first and last week (for these charts)

df_ = df[(df.week_start > df.week_start.min()) & (df.week_start < df.week_start.max())].copy()


df_.groupby('week_start').visits.sum().plot()


df_.groupby('week_start').transactions.sum().plot()


# Next we'll try to forecast this trend forward. It will be difficult, since there's only one year of data to train models on.

# ## Forcasting with Facebook Prophet

from fbprophet import Prophet
import warnings
# Ignore warnings from prophet lib
warnings.filterwarnings('ignore', 'Conversion of the second argument of issubdtype')


# ### Transactions

# Making sure to add `yearly_seasonality` as we expect yearly periodicity e.g. Christmas, black friday, etc...
# 
# Tune the parameter values (`weekly_seasonality` and `yearly_seasonality` and `seasonality_prior_scale`) to fit the given data. Higher values will lead to tighter fits (be careful about overfitting here).

df_prophet = df[['date', 'transactions']]    .rename(columns={'date': 'ds', 'transactions': 'y'})
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet(
    daily_seasonality=False,
    weekly_seasonality=1,
    yearly_seasonality=20,
    seasonality_prior_scale=0.1,
)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)
fig = m.plot(forecast)
plt.ylabel('Transactions')
plt.xlabel('Date')
savefig('sales_forecast_transactions')


fig = m.plot_components(forecast)
fig.axes[0].set_xlabel('Date')
fig.axes[0].set_ylabel('Overall Trend')
fig.axes[1].set_ylabel('Weekly Trend')
fig.delaxes(fig.axes[2])
savefig('sales_forecast_transactions_trends')


# ### Sales

df_prophet = df[['date', 'transactionRevenue']]    .rename(columns={'date': 'ds', 'transactionRevenue': 'y'})
df_prophet['y'] = df_prophet['y'] / 1e6
df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

m = Prophet(
    daily_seasonality=False,
    weekly_seasonality=1,
    yearly_seasonality=20,
    seasonality_prior_scale=0.1,
)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)
fig = m.plot(forecast)

savefig('sales_forecast')


fig = m.plot_components(forecast)
fig.axes[0].set_xlabel('Date')
fig.axes[0].set_ylabel('Overall Trend')
fig.axes[1].set_ylabel('Weekly Trend')
fig.delaxes(fig.axes[2])
savefig('sales_forecast_trends')


# ### Predictions by quarter

forecast['date'] = pd.to_datetime(forecast['ds'])


def add_quarters(df):
    """
    Add labels for quarter number. May need to adjust for specific
    fiscal calendar. Only covers until end of 2018.
    
    df : pd.DataFrame
        Should have "date" column that is datetime dtype.
    """
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


# For the Google Merch Store, we only have partial data for 2016 & 2018 Q3, so we'll want to filter these out.

m_google_merch = (forecast.quarter != '2016 Q3') & (forecast.quarter != '2018 Q3')

s_transactionRevenue = (
    forecast[m_google_merch].groupby(['quarter_num', 'quarter'])
        .yhat.sum().reset_index()
        .set_index('quarter_num').sort_index(ascending=True)
        .set_index('quarter')['yhat']
)

# Print it
s_transactionRevenue.apply(lambda x: '${:,}'.format(round(x)))


# Display these results in a dataframe

# Get the actual sales (as opposed to predicted above)

df = add_quarters(df)
m_google_merch = (df.quarter != '2016 Q3') & (df.quarter != '2018 Q3')
s_actual_transactionRevenue = (
    df[m_google_merch].groupby(['quarter_num', 'quarter'])
        .transactionRevenue.sum().reset_index()
        .set_index('quarter_num').sort_index(ascending=True)
        .set_index('quarter')['transactionRevenue'] / 1e6
)

# Print it
s_actual_transactionRevenue.apply(lambda x: '${:,}'.format(round(x)))


# Generate a YoY table view

forecast_results = pd.DataFrame({
    'Reporting Period': ['Q4', 'Q1', 'Q2'],
    'Prev Year': [
        s_actual_transactionRevenue[s_actual_transactionRevenue.index=='2016 Q4'].values[0] if (s_actual_transactionRevenue.index=='2016 Q4').sum() else 0,
        s_actual_transactionRevenue[s_actual_transactionRevenue.index=='2017 Q1'].values[0] if (s_actual_transactionRevenue.index=='2017 Q1').sum() else 0,
        s_actual_transactionRevenue[s_actual_transactionRevenue.index=='2017 Q2'].values[0] if (s_actual_transactionRevenue.index=='2017 Q2').sum() else 0,
    ],
    'Forecasted': [
        s_transactionRevenue[s_transactionRevenue.index=='2017 Q4'].values[0],
        s_transactionRevenue[s_transactionRevenue.index=='2018 Q1'].values[0],
        s_transactionRevenue[s_transactionRevenue.index=='2018 Q2'].values[0],
    ],
})
forecast_results['YoY (%)'] = ((forecast_results['Forecasted'] - forecast_results['Prev Year'])
                                / forecast_results['Prev Year'] * 100).apply(lambda x: '{:+.0f}%'.format(x))
forecast_results['Prev Year'] = forecast_results['Prev Year'].apply(lambda x: '${:,.0f}'.format(x))
forecast_results['Forecasted'] = forecast_results['Forecasted'].apply(lambda x: '${:,.0f}'.format(x))
forecast_results.set_index('Reporting Period', inplace=True)
forecast_results.to_csv('../../results/tables/sales_forecast.csv')


forecast_results


# ## Forecasting by device, region, source, etc...
# 
# The Google Merch Store dataset in BigQuery gives limited access to segmentation variables. Some interesting options we do have access to are device, region and traffic source.

# ### Read from BigQuery
# 
# Run this cell to read from BigQuery.

get_ipython().run_cell_magic('time', '', '"""\nUsing bigquery\n"""\n\nfrom google.cloud import bigquery\n# Using environment variables to authenticate\nclient = bigquery.Client()\n\ndef pull_daily_data(verbose=False):\n    dataset = client.get_dataset(\'bigquery-public-data.google_analytics_sample\')\n\n    data = []\n    for table in tqdm_notebook(list(client.list_tables(dataset))):\n        if verbose:\n            print(\'Querying {}\'.format(table.table_id))\n        query_job = client.query(\'\'\'\n            select\n              date,\n              sum(totals.visits),\n              sum(totals.pageviews),\n              sum(totals.transactions),\n              sum(totals.transactionRevenue),\n              trafficSource.source as source,\n              device.deviceCategory as deviceCategory,\n              geoNetwork.country as country,\n              geoNetwork.region as region\n            from `bigquery-public-data.google_analytics_sample.{}`\n            group by date, source, deviceCategory, country, region;\n        \'\'\'.format(table.table_id))\n        results = query_job.result().to_dataframe()\n        results.columns = [\'date\', \'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\',\n                           \'source\', \'deviceCategory\', \'country\', \'region\']\n        data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df\n\nbq_results = pull_daily_data()')


bq_results.head()


df = bq_results.copy()
df.date = pd.to_datetime(df.date)

f_path = os.path.join(os.getenv('DATA_PATH'), 'interim', 'sales_forecast_segments_raw.csv')
if os.path.isfile(f_path):
    raise Exception(
        'File exists! Run line below in separate cell to overwrite it. '
        'Otherwise just run cell below to load file.')

df.to_csv(f_path, index=False)


# ### Load pre-queried data

def load_file(f_path):
    if not os.path.exists(f_path):
        raise Exception('No data found. Run data load script above.')
    print('Loading {}'.format(f_path))
    df = pd.read_csv(f_path)
    df.date = pd.to_datetime(df.date)
    return df

f_path = os.path.join(os.getenv('DATA_PATH'), 'interim', 'sales_forecast_segments_raw.csv')
df = load_file(f_path)


df.head()


df['week'] = df.date.apply(lambda x: x.strftime('%W'))
df['year'] = df.date.apply(lambda x: x.strftime('%Y'))
df['week_start'] = df[['week', 'year']].apply(
    lambda x: datetime.datetime.strptime('{}-{}-1'.format(x.year, x.week), '%Y-%W-%w'),
    axis=1
)


df.dtypes


df_ = df[(df.week_start > df.week_start.min()) & (df.week_start < df.week_start.max())].copy()


df_.groupby('week_start').visits.sum().plot()


df_.groupby('week_start').transactions.sum().plot()


# Next we'll try to forecast this trend forward. It will be difficult, since there's only one year of data to train models on.

# ### Forcasting with Transactions

from fbprophet import Prophet
import warnings
# Ignore warnings from prophet lib
warnings.filterwarnings('ignore', 'Conversion of the second argument of issubdtype')


# Making sure to add `yearly_seasonality` as we expect yearly periodicity e.g. Christmas, black friday, etc...
# 
# Tune the parameter values (`weekly_seasonality` and `yearly_seasonality` and `seasonality_prior_scale`) to fit the given data. Higher values will lead to tighter fits (be careful about overfitting here).

get_ipython().run_line_magic('pinfo', 'Prophet.plot')


def build_prophet_df(df, segment_col):
    df_prophet = (
        df.groupby(['date', segment_col]).transactions.sum()
        .reset_index()
        .rename(columns={'transactions': 'y'})
    )
    df_prophet['ds'] = df_prophet['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df_prophet = df_prophet.sort_values('date', ascending=True)
    return df_prophet

def segment_forecast_daily(
    df,
    segment_col,
    max_num_segments=6,
    daily_seasonality=False,
    weekly_seasonality=2,
    yearly_seasonality=10,
    seasonality_prior_scale=0.1,
):
    df_prophet = build_prophet_df(df, segment_col)
    
    segs = (
        df_prophet.groupby(segment_col)['y'].sum()
        .sort_values(ascending=False)
        .head(max_num_segments).index.tolist()
    )
    models = []
    forecasts = []
    for seg in segs:
        m = Prophet(
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            seasonality_prior_scale=seasonality_prior_scale,
        )
        m.fit(df_prophet[df_prophet[segment_col] == seg])
        models.append(m)
        
        future = m.make_future_dataframe(periods=365, freq='D')
        forecast = m.predict(future)
        forecasts.append(forecast)
        
    colors = list('rbgykmc') * 10
    fig, axes = plt.subplots(len(forecasts), sharex=True)
    for i, (ax, seg, forecast, c) in enumerate(zip(axes, segs, forecasts, colors)):
        forecast.set_index('ds')['yhat'].plot(label=seg, ax=ax, color=c)
        (df_prophet[df_prophet[segment_col] == seg]
         .set_index('date')['y'].plot(label='_', ax=ax, color=c, marker='o', linewidth=0))
        ax.legend()
        if i == int(len(forecasts) / 2):
            ax.set_ylabel('Transactions')
        else:
            ax.set_ylabel('')

    plt.xlabel('')
    savefig('sales_forecast_{}_transactions'.format('-'.join(segment_col.split(' '))))
    plt.show()

    for seg, m, forecast, c in zip(segs, models, forecasts, colors):
        fig = m.plot_components(forecast)
        fig.axes[0].set_xlabel('{} - Date'.format(seg.title()))
        fig.axes[0].set_ylabel('Overall Trend')
        fig.axes[0].lines[0].set_color(c)
        fig.axes[1].set_xlabel('{} - Day of week'.format(seg.title()))
        fig.axes[1].set_ylabel('Weekly Trend')
        fig.axes[1].lines[0].set_color(c)
        fig.delaxes(fig.axes[2])
        savefig('sales_forecast_{}={}_transactions_trends'.format(
            '-'.join(segment_col.split(' ')),
            '-'.join(seg.split(' '))
        ))
        plt.show()


segment_forecast_daily(df, 'deviceCategory')


# Cool! We can see distinct trends for day of week on mobile VS desktop.

df_fltr = df[~(df.source.isin(['not available in demo dataset', '(not set)']))]
segment_forecast_daily(df_fltr, 'source')


df_fltr = df[~(df.country.isin(['not available in demo dataset', '(not set)']))]
segment_forecast_daily(df_fltr, 'country')


df_fltr = df[~(df.region.isin(['not available in demo dataset', '(not set)']))]
segment_forecast_daily(df_fltr, 'region')


# ## Forecasting by product

# ### Read from bigquery

get_ipython().run_cell_magic('time', '', '"""\nUsing bigquery\n"""\n\ndef pull_daily_product_sales(verbose=False):\n    dataset = client.get_dataset(\'bigquery-public-data.google_analytics_sample\')\n\n    data = []\n    for table in tqdm_notebook(list(client.list_tables(dataset))):\n        if verbose:\n            print(\'Querying {}\'.format(table.table_id))\n        query_job = client.query(\'\'\'\n        SELECT\n            h.item.productName AS other_purchased_products,\n            COUNT(h.item.productName) AS quantity\n        FROM `bigquery-public-data.google_analytics_sample.{}`,\n            UNNEST(hits) as h\n        WHERE (\n            fullVisitorId IN (\n                SELECT fullVisitorId\n                FROM `bigquery-public-data.google_analytics_sample.{}`,\n                    UNNEST(hits) as h\n                WHERE h.item.productName CONTAINS \'Product Item Name A\'\n                AND totals.transactions>=1\n                GROUP BY fullVisitorId\n            )\n            AND h.item.productName IS NOT NULL\n            AND h.item.productName != \'Product Item Name A\'\n        )\n        GROUP BY other_purchased_products\n        ORDER BY quantity DESC;\n        \'\'\'.format(table.table_id, table.table_id))\n        results = query_job.result().to_dataframe()\n        results.columns = [\'date\', \'visits\', \'pageviews\', \'transactions\', \'transactionRevenue\']\n        data.append(results)\n\n    df = pd.concat(data, ignore_index=True, sort=False)\n    return df\n\nbq_product_results = pull_daily_product_sales()')


bq_product_results.head()


df = bq_product_results.copy()
df.date = pd.to_datetime(df.date)

f_path = os.path.join(os.getenv('DATA_PATH'), 'interim', 'sales_forecast_by_product_raw.csv')
if os.path.isfile(f_path):
    raise Exception(
        'File exists! Run line below in separate cell to overwrite it. '
        'Otherwise just run cell below to load file.')

df.to_csv(f_path, index=False)


# ### Read from `jsonl` local

from typing import List, Tuple, Dict


get_ipython().run_cell_magic('time', '', '"""\nUsing local jsonl\n"""\nERRORS = []\n\ndef pull_daily_product_sales(\n    verbose=False,\n    raise_errors=False,\n    test=False,\n) -> Tuple[pd.DataFrame, dict]:\n    if os.getenv(\'DATA_PATH\') is None:\n        raise ValueError(\'Please set environment variable DATA_PATH\')\n\n    dataset = sorted(glob.glob(os.path.join(os.getenv(\'DATA_PATH\'), \'raw\', \'*.jsonl\')))\n    print(\'Loading from {}, etc...\'.format(\', \'.join(dataset[:3])))\n\n    data = []\n    for table in tqdm_notebook(dataset):\n        if verbose:\n            print(\'Scanning {}\'.format(table))\n        with open(table, \'r\') as f:\n            for line in f:\n                d = json.loads(line)\n                try:\n                    if not d[\'totals\'][\'transactions\']:\n                        # No purchases, continue to next visitor\n                        continue\n                    for hit in d[\'hits\']:\n                        for product in hit[\'product\']:\n                            if product[\'productRevenue\']:\n                                data.append({\n                                    \'date\': d[\'date\'],\n                                    \'visitId\': d[\'visitId\'],\n                                    \'fullVisitorId\': d[\'fullVisitorId\'],\n                                    \'product\': product,\n                                })\n                except Exception as e:\n                    if verbose:\n                        print(\'Error raised when reading row:\\n{}\'.format(e))\n                    ERRORS.append([table, e])\n                    if raise_errors:\n                        raise(e)\n                        \n        if test and (table == dataset[1]):\n            break\n\n    cols_main = [\'date\', \'visitId\', \'fullVisitorId\']\n    cols_product = [\n        \'productSKU\', \'v2ProductName\', \'v2ProductCategory\', \'productVariant\',\n        \'productRevenue\', \'productQuantity\', \'productRefundAmount\'\n    ]\n    df_data = [\n        [d.get(col, float(\'nan\')) for col in cols_main]\n        + [d[\'product\'].get(col) for col in cols_product]\n        for d in data\n    ]\n    df = pd.DataFrame(df_data, columns=(cols_main+cols_product))\n    return df, data\n\njsonl_product_results, nosql_data = pull_daily_product_sales(raise_errors=True, test=True)')


jsonl_product_results, nosql_data = pull_daily_product_sales()


df = jsonl_product_results.copy()
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


# ### Exploration

# Top selling products

df.v2ProductName.value_counts(ascending=False).head(10)


# Top selling products by revenue

(df.groupby('v2ProductName').productRevenue.sum() / 1e6).sort_values(ascending=False).head(10)


# How many of the *same* product are ordered together?

s = df.productQuantity.value_counts().sort_index(ascending=False)
s.name = 'Number of Product Duplicates in Baskets'
pd.DataFrame(index=list(range(2,21))).join(s).iloc[::-1].plot.barh(color='b')
savefig('sales_forecast_product_duplicates')


# Most common groups of 2

s = df[df.productQuantity == 2].groupby(['v2ProductName', 'productVariant']).size().sort_values(ascending=False)
s.name = 'Counts'
s.reset_index().set_index('Counts').head()


# Most common groups of 3

s = df[df.productQuantity == 3].groupby(['v2ProductName', 'productVariant']).size().sort_values(ascending=False)
s.name = 'Counts'
s.reset_index().set_index('Counts').head()


# How many _different_ products are purchased together? i.e. final basket sizes

# Add a column for transaction ID
# See docs for more details on this: https://support.google.com/analytics/answer/3437719?hl=en
df['transactionId'] = df['visitId'].astype(str) + '|' + df['fullVisitorId'].astype(str)


m = ~(df[['transactionId', 'v2ProductName', 'productVariant']].duplicated())
s = df[m].groupby('transactionId').size().value_counts().sort_index(ascending=False)
s.name = 'Number of Unique Products in Baskets'
pd.DataFrame(index=list(range(1,21))).join(s).iloc[::-1].plot.barh(color='b')
savefig('sales_forecast_unique_basket_products')


# Which 2 items are most commonly purchased together?

fig = plt.figure(figsize=(4, 2))

m = ~(df[['transactionId', 'v2ProductName', 'productVariant']].duplicated())
s = df[m].groupby('transactionId').size()
transaction_ids = s[s == 2].index.tolist()

pairs = []
for transaction_id in tqdm_notebook(transaction_ids):
    data = df[df.transactionId == transaction_id][['v2ProductName', 'productVariant']].values.tolist()
    if len(data) != 2:
        continue
    
    pair = ', '.join(['{} ({})'.format(d[0].strip(), d[1].strip()) for d in data])
    pairs.append(pair)
    
s = pd.Series(pairs).value_counts(ascending=False)

s.head().iloc[::-1].plot.barh(color='b')

savefig('sales_forecast_product_pairs_1')


# Same chart but ignoring product variants

fig = plt.figure(figsize=(4, 2))

m = ~(df[['transactionId', 'v2ProductName']].duplicated())
s = df[m].groupby('transactionId').size()
transaction_ids = s[s == 2].index.tolist()

pairs = []
for transaction_id in tqdm_notebook(transaction_ids):
    data = df[df.transactionId == transaction_id]['v2ProductName'].values.tolist()
    if len(data) != 2:
        continue
    
    pair = ', '.join(data)
    pairs.append(pair)
    
s = pd.Series(pairs).value_counts(ascending=False)

s.head().iloc[::-1].plot.barh(color='b')

savefig('sales_forecast_product_pairs_2')


# ### Forecasts by product

# In general it's difficult to make granular predictions like this on a daily basis. For this reason we will group by week.
# 
# In order to help accurately account for anomaly events (e.g. someone places an order for 100 hoddies), I am going to remove outliers (by computing standard deviation and filtering out e.g. `>2s.d.`). Then I'll take that missing revenue and add it back into the quarterly predictions, splitting evenly.

def product_forcast(
    df,
    product_name,
    ignore_std=2,
    yearly_seasonality=10,
    weekly_seasonality=False,
    daily_seasonality=False,
    add_back_outlier_revenue=True,
) -> pd.DataFrame:

    # Filter on product & remove outliers
    df_ = df[df.v2ProductName == product_name].copy()
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
    
    # Group by week
    df_ = (df_[m]
           .groupby(pd.Grouper(key='date', freq='W-MON'))
           .productRevenue.sum().reset_index())

    df_prophet = df_[['date', 'productRevenue']]        .rename(columns={'date': 'ds', 'productRevenue': 'y'})
    df_prophet['y'] = df_prophet['y'] / 1e6
    df_prophet['ds'] = df_prophet['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

    model = Prophet(yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=52, freq='7D')
    forecast = model.predict(future)
    model.plot(forecast)
    savefig('sales_forecast_{}'.format(slugify(product_name)))
    
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
        df_[m_q].groupby(['quarter_num', 'quarter'])
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
    forecast_results.to_csv('../../results/tables/sales_forecast_{}.csv'.format(slugify(product_name)))
    
    return forecast, forecast_results, fig


forecast, forecast_results, fig = product_forcast(
    df,
    'Google Men\'s  Zip Hoodie',
    yearly_seasonality=8,
)
plt.show()
forecast_results


# Run for each of the top 10 Top selling products

for product in df.v2ProductName.value_counts(ascending=False).head(10).index.tolist():
    print('-'*20)
    print(Fore.RED + product + Style.RESET_ALL)
    forecast, forecast_results, fig = product_forcast(
        df,
        product,
        yearly_seasonality=8,
    )
    plt.show()
    display(forecast_results)
    print()


from IPython.display import HTML
HTML('<style>div.text_cell_render{font-size:130%;padding-top:50px;padding-bottom:50px}</style>')

