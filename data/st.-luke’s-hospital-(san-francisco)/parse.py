#!/usr/bin/env python

import os
from glob import glob
import json
import codecs
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

here = os.path.dirname(os.path.abspath(__file__))
folder = os.path.basename(here)
latest = '%s/latest' % here
year = datetime.datetime.today().year

output_data = os.path.join(here, 'data-latest.tsv')
output_year = os.path.join(here, 'data-%s.tsv' % year)

# Don't continue if we don't have latest folder
if not os.path.exists(latest):
    print('%s does not have parsed data.' % folder)
    sys.exit(0)

# Don't continue if we don't have results.json
results_json = os.path.join(latest, 'records.json')
if not os.path.exists(results_json):
    print('%s does not have results.json' % folder)
    sys.exit(1)

with open(results_json, 'r') as filey:
    results = json.loads(filey.read())

columns = ['charge_code', 
           'price', 
           'description', 
           'hospital_id', 
           'filename', 
           'charge_type']

df = pd.DataFrame(columns=columns)

# First parse standard charges (doesn't have DRG header)
for result in tqdm(results):
    filename = os.path.join(latest, result['filename'])
    if not os.path.exists(filename):
        print('%s is not found in latest folder.' % filename)
        continue

    if os.stat(filename).st_size == 0:
        print('%s is empty, skipping.' % filename)
        continue

    charge_type = 'standard'

    print("Parsing %s" % filename)

    if filename.endswith('json'):

         with codecs.open(filename, "r", 
            encoding='utf-8-sig', errors='ignore') as filey:
             content = json.load(filey)

         charge_types = {'DRG': 'drg', 
                         'IP': 'inpatient', 
                         'OP': 'outpatient', 
                         'RX': 'pharmacy', 
                         'SUP': 'supply'}

         # Create dataframe and make empty strings null
         temp = pd.DataFrame.from_records(content['CDM']).replace({'': np.nan})

         # For any null hospital names, 
         # replace with hospital_id from records.json
         temp['HOSPITAL_NAME'].fillna(result["hospital_id"], inplace=True)

         # Rename columns accordingly
         # Note that we'll need to fill in the filename column manually
         temp.rename(columns={
            'CDM': 'charge_code',
            'CHARGE': 'price',
            'DESCRIPION': 'description',
            'HOSPITAL_NAME': 'hospital_id', 
            'SERVICE_SETTING': 'charge_type'
            }, inplace=True)

         # Reformat hospital_id to be lowercase with hyphens instead of spaces
         temp['hospital_id'] = \
         temp['hospital_id'].astype(str).str.lower().str.replace(" ", "-")

         # Make sure we map the values of charge_type properly
         temp['charge_type'] = temp['charge_type'].map(charge_types)

         # Add filename column
         temp['filename'] = result['filename']

         # Remove extraneous columns
         extra_cols = list(temp.columns[~temp.columns.isin(columns)])
         temp.drop(columns=extra_cols, inplace=True)

         df = df.append(temp, ignore_index=True)

    else:
        break


# Remove empty rows
df = df.dropna(how='all')
print(df.head())

# Save data! ...and make sure columns are ordered in standard way
print(df.shape)
df.to_csv(output_data, columns=columns, sep='\t', index=False)
df.to_csv(output_year, columns=columns, sep='\t', index=False)
