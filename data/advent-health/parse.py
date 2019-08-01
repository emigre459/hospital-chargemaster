#!/usr/bin/env python

import os
import xmltodict
from glob import glob
import json
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm # progress bar

here = os.path.dirname(os.path.abspath(__file__))
folder = os.path.basename(here)

year = datetime.datetime.today().year
output_data = os.path.join(here, 'data-latest.tsv')
output_year = os.path.join(here, 'data-%s.tsv' % year)

latest = '%s/latest' % here

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

# Helper Functions - different formats of XML
def process_dataroot(content, df, hospital_id, filename):

    # Hospital name is the key that doesn't start with @
    for hospital_name in content['dataroot'].keys():
        if not hospital_name.startswith('@'):            
            break

    temp = pd.DataFrame(content['dataroot'][hospital_name])\
    .rename(columns={'Charge_x0020_Code': 'charge_code',
        'Description': 'description'})

    # Need to melt the dataframe so that 
    # unique price type columns (e.g. Inpatient vs. Outpatient)
    # are captured in a charge_type field
    temp = temp.melt(id_vars=['charge_code', 'description'],
        value_name='price', var_name='charge_type')\
    .dropna(subset=['price'])

    # Map charge_type values to standard labels
    temp['charge_type'] = \
    temp['charge_type'].map({'Inpatient_x0020_Price': 'inpatient',
                             'Outpatient_x0020_Price': 'outpatient'})

    # add in filename and hospital_id columns
    temp['filename'] = filename
    temp['hospital_id'] = hospital_id

    return df.append(temp, ignore_index=True, sort=True)


def process_workbook(df, hospital_id, filename, filepath):    
    # First row is header
    temp = pd.read_csv(filepath, thousands=',')

    # Rename to standard column names
    temp.rename(columns={'Charge Description': 'description',
        'Charge Amount': 'price'}, inplace=True)

    temp['charge_code'] = np.nan
    temp['hospital_id'] = hospital_id
    temp['filename'] = filename
    temp['charge_type'] = 'standard'

    return df.append(temp, ignore_index=True, sort=True)


seen = []
for result in tqdm(results):
    filename = os.path.join(latest, result['filename'])
    if not os.path.exists(filename):
        print('%s is not found in latest folder.' % filename)
        continue

    if os.stat(filename).st_size == 0:
        print('%s is empty, skipping.' % filename)
        continue

    if result['filename'] in seen:
        continue

    seen.append(result['filename'])
    print('Parsing %s' % filename)

    # Parse the different XML files
    if filename.endswith('xml'):
        with open(filename, 'r') as filey:
            content = xmltodict.parse(filey.read())

        if "dataroot" in content:
            df = process_dataroot(content, df, result['uri'], 
                result['filename'])
    elif filename.endswith('csv'):
        df = process_workbook(df, result['uri'], 
            result['filename'], filename)

    else:
        break

    # Save data as we go
    print(df.head(3))
    df.to_csv(output_data, columns=columns, sep='\t', index=False)
    df.to_csv(output_year, columns=columns, sep='\t', index=False)
