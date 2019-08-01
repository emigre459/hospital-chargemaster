#!/usr/bin/env python

import os
from glob import glob
import json
import pandas
import datetime
import tqdm

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

df = pandas.DataFrame(columns=columns)

# First parse standard charges (doesn't have DRG header)
for result in results:
    filename = os.path.join(latest, result['filename'])
    if not os.path.exists(filename):
        print('%s is not found in latest folder.' % filename)
        continue

    if os.stat(filename).st_size == 0:
        print('%s is empty, skipping.' % filename)
        continue

    if filename.endswith('xlsx'):
        content = pandas.read_excel(filename)
    else:
        break
 
    print("Parsing %s" % filename)
    print(f"Original shape of data is {content.shape}")


    # Update by row
    # 'FACILITY', 
    # 'CMS_PROV_ID', 
    # 'HOSPITAL_NAME', 
    # 'SERVICE_SETTING', 
    # 'CDM', 
    # 'DESCRIPION', 
    # 'REVENUE_CODE', 
    # 'CHARGE']

    # charge type
    content['SERVICE_SETTING'] = content['SERVICE_SETTING'].map({
        'IP': 'inpatient',
        'OP': 'outpatient',
        'DRG': 'drg',
        'SUP': 'supply',
        'RX': 'pharmacy'
    })

    print(f"Shape of data after mapping is {content.shape}")

    # If revenue code is null then charge code is CDM value, 
    # otherwise it's revenue code
    content['REVENUE_CODE'].fillna(content['CDM'], inplace = True)
    print(f"Shape of data after fillna is {content.shape}")

    # Make sure hospital name is to project's standard ID format
    content['HOSPITAL_NAME'] = \
    content['HOSPITAL_NAME'].astype(str).str.lower().str.replace(" ", "-")

    print(f"Shape of data after hospital id formatting is {content.shape}")

    temp = pandas.DataFrame(data = [
        content['REVENUE_CODE'],     # charge code
        content['CHARGE'],           # price
        content['DESCRIPION'],       # description
        content['HOSPITAL_NAME'],    # hospital id
        pandas.Series([result['filename']] * len(content['HOSPITAL_NAME'])), # filename
        content['SERVICE_SETTING']   # charge type
        ]).transpose()

    temp.columns = columns

    df = df.append(temp, ignore_index=True)

print(f"Shape of df before dropping nulls is {df.shape}\n")
print(df.head())
# Remove empty rows
df.dropna(how='all', inplace=True)

# Save data!
print(df.shape)
df.to_csv(output_data, sep='\t', index=False)
df.to_csv(output_year, sep='\t', index=False)
