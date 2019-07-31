#!/usr/bin/env python

import os
from glob import glob
import json
import pandas
import datetime

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
 
    print("Parsing %s" % filename)


    # Update by row
    # 'FACILITY', 
    # 'CMS_PROV_ID', 
    # 'HOSPITAL_NAME', 
    # 'SERVICE_SETTING', 
    # 'CDM', 
    # 'DESCRIPION', 
    # 'REVENUE_CODE', 
    # 'CHARGE']
    for row in content.iterrows():

        if row[1].SERVICE_SETTING == 'IP':
            charge_type = 'inpatient'
        elif row[1].SERVICE_SETTING == 'OP':
            charge_type = 'outpatient'
        elif row[1].SERVICE_SETTING == 'DRG':
            charge_type = 'drg'
        elif row[1].SERVICE_SETTING == 'SUP':
            charge_type = 'supply'
        elif row[1].SERVICE_SETTING == 'RX':
            charge_type = 'pharmacy'
        
        # Charge code can be revene code or CDM ?
        charge_code = row[1].REVENUE_CODE
        if pandas.isnull(charge_code):
            charge_code = row[1].CDM

        idx = df.shape[0] + 1
        entry = [charge_code,                 # charge code
                 row[1]["CHARGE"],            # price
                 row[1]['DESCRIPTION'],        # description
                 row[1]["HOSPITAL_NAME"],     # hospital_id
                 result['filename'],
                 charge_type] 
        df.loc[idx,:] = entry


# Remove empty rows
df = df.dropna(how='all')

# Save data!
print(df.shape)
df.to_csv(output_data, sep='\t', index=False)
df.to_csv(output_year, sep='\t', index=False)
