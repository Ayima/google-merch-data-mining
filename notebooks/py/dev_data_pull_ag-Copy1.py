
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

# ### Pulling the data
# 
# > The first 1 TB of query data processed per month is free

from google.cloud import bigquery


from dotenv import load_dotenv
load_dotenv('../../.env')


client = bigquery.Client()


query_job = client.query("""
    SELECT
      CONCAT(
        'https://stackoverflow.com/questions/',
        CAST(id as STRING)) as url,
      view_count
    FROM `bigquery-public-data.stackoverflow.posts_questions`
    WHERE tags like '%google-bigquery%'
    LIMIT 10
""")

results = query_job.result()  # Waits for job to complete.


results


for row in results:
    print("{} : {} views".format(row.url, row.view_count))


# Let's select some GA data

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


len(data)


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

# def dump_data(table):
#     bucket_name = 'bigquery-public-data-ga-sample'
#     destination_uri = "gs://{}/{}.json".format(bucket_name, table.table_id)
#     extract_job = client.extract_table(
#         table,
#         destination_uri,
#         # Location must match that of the source table.
#         location="US",
#     )  # API request
#     extract_job.result()  # Waits for job to complete.


def dump_data(table_id, limit=None):
    query_job = client.query("""
        SELECT *
        FROM `bigquery-public-data.google_analytics_sample.{}`
    """.format(table_id))
    if limit is not None:
        query_job += '\nLIMIT {}'.format(limit)
    results = query_job.result()  # Waits for job to complete.
    
    with open('../../data/raw/{}.jsonl'.format(table_id), 'w') as f:
        for result in results:
            f.write('{}\n'.format(
                json.dumps(dict(result.items()))
            ))


dataset_id = 'bigquery-public-data.google_analytics_sample'
dataset = client.get_dataset(dataset_id)
tables = list(client.list_tables(dataset))
print('got {} tables'.format(len(tables)))

for table in tables:
    print('dumping {}'.format(table.table_id))
    dump_data(table.table_id)
    break


ls ../../data/raw/


cat ../../data/raw/ga_sessions_20160801.jsonl | wc -l


# ### Saving the data locally
# 
# Pull all the data (~25gigs | 3 hours)

dataset_id = 'bigquery-public-data.google_analytics_sample'
dataset = client.get_dataset(dataset_id)
tables = list(client.list_tables(dataset))
print('got {} tables'.format(len(tables)))

for table in tables:
    print('dumping {}'.format(table.table_id))
    dump_data(table.table_id)


from IPython.display import HTML
HTML('<style>div.text_cell_render{font-size:130%;padding-top:50px;padding-bottom:50px}</style>')

