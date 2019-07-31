[![DOI](https://zenodo.org/badge/166080583.svg)](https://zenodo.org/badge/latestdoi/166080583)

# Hospital Chargemaster Analysis and Modeling

This is an analysis of publicly-available hospital chargemasters that uses the the [Dinosaur Dataset](https://vsoch.github.io/datasets/2019/hospital-chargemasters/) as its starting point. 

### Table of Contents

1. [The Data](#data)
	1. [Chargemaster Data Collection Process](#data_collection)
	2. [Data Beyond Chargemasters](#extra_features)


## What are these data? <a name="data"></a>

As of January 1, 2019, U.S. hospitals (or at least all of those receiving Medicare and Medicaid funding, which is likely all of them) are required to share their price lists for all of their consumables (e.g. 
hypodermic needles) and procedures (e.g. triple-bypass surgery). However,
 it remains a problem that the data released
[is not intended for human consumption](https://qz.com/1518545/price-lists-for-the-115-biggest-us-hospitals-new-transparency-law/). 

This repo does the following:

1. Takes the chargemaster data originally gathered by @vsoch in early 2019
2. Pushes all of the different chargemasters into a single DataFrame (data table)
3. Cleans it all up
	* The formats across chargemasters differ quite a bit at times, leading to bad initial parsing
4. Augments the chargemaster data with hospital metadata for analytical purposes
5. Links everything together via an sqlite database

### How do the chargemaster data get gathered? <a name="data_collection"></a>

#### 1. Get List of Hospital Pages

There is a list of hospitals and chargemaster URLs in the [hospitals.tsv](hospitals.tsv) 
file, generated via the [0.get_hospitals.py](0.get_hospitals.py) script. 
The file includes the following variables, separated by tabs:

 - **hospital_name** is the human friendly name
 - **hospital_url** is the human friendly URL, typically the page that includes a link to the data.
 - **hospital_id** is the unique identifier for the hospital, the hospital name, in lowercase, with spaces replaced with `-`

This represents the original set of hospitals that I obtained from a [compiled list](https://qz.com/1518545/price-lists-for-the-115-biggest-us-hospitals-new-transparency-law/), and is kept
for the purpose of keeping the record.

#### 2. Organize Data

Each hospital has records kept in a subfolder in the [data](data) folder. Specifically,
each subfolder is named according to the hospital name (made all lowercase, with spaces 
replaced with `-`). If a subfolder begins with an underscore, it means that I wasn't
able to find the charge list on the hospital site (and maybe you can help?) 
Within that folder, you will find:

 - `scrape.py`: A script to scrape the data
 - `browser.py`: If we need to interact with a browser, we use selenium to do this.
 - `latest`: a folder with the last scraped (latest data files)
 - `YYYY-MM-DD` folders, where each folder includes:
   - `records.json` the complete list of records scraped for a particular data
   - `*.csv` or `*.xlsx` or `*.json`: the scraped data files.

The first iteration was run locally (to test the scraping). One significantly different
scraper is the [oshpd-ca](data/oshpd-ca) folder, which includes over 795 hospitals! Way to go
California! Additionlly, [avent-health](data/advent-health) provides (xml) charge lists
for a ton of states.

#### 3. Parsing

This is likely one of the hardest steps. I wanted to see the extent to which I could
create a simple parser that would generate a single TSV (tab separted value) file
per hospital, with minimally an identifier for a charge, and a price in dollars. If
provided, I would also include a description and code:

 - **charge_code**
 - **price**
 - **description**
 - **hospital_id**
 - **filename**

Each of these parsers is also in the hospital subfolder, and named as "parser.py." The parser would output a data-latest.tsv file at the top level of the folder, along with a dated (by year `data-<year>.tsv`). At some point
I realized that there were different kinds of charges, including inpatient, outpatient, DRG (diagnostic related group) and others called
"standard" or "average." I then went back and added an additional column
to the data:

 - **charge_type** can be one of standard, average, inpatient, outpatient, drg, or (if more detail is supplied) insured, uninsured, pharmacy, or supply. This is not a gold standard labeling but a best effort. If not specified, I labeled as standard, because this would be a good assumption.

## Data Beyond Chargemasters <a name="extra_features"></a>

In order to provide enough features/variables at the hospital level to build predictive models for different procedure and consumable costs, hospital-level datasets were merged into the chargemaster data. Here I provide some brief descriptions for the versioned datasets used in this work. More information at a table and column level can be found in [the data dictionary](data_dictionary.csv). All of these datasets were accessed programmatically using [the data.medicare.gov APIs for the Hospital Compare datasets](https://data.medicare.gov/data/hospital-compare). Whenever possible, the version of the originating dataset (e.g. it's most recent metadata update) is provided in [the data dictionary](data_dictionary.csv).

1. `Hospital_Readmissions_Reduction_Program`
	* Provides data on individual hospitals' predictions of readmissions for patients within 30 days that are suffering from certain types of ailments and received certain kinds of treatments. Effectively, it provides information on how good individual hospitals are at predicting patient readmissions as compared to a similar hospital's average.
	* The `Provider ID` column of these data was used to give each hospital in the dataset a unique identifier by matching to the `Hospital Name` field as best as possible.
	
2. `Skilled_Nursing_Facility_Quality_Reporting_Program_07-24-2019.csv`
	* Dataset last updated on 7/24/2019, but data actually cover 10/1/2017 to 9/30/2018
	* Provides data about Skilled Nursing Facilities (which can be hospitals that provide long-term nursing-home-like care, or standalone nursing homes), in particular a `Score` column that rates the quality of long-term care provided at that facility.
	* Fair warning: lots of missing values in this one
	* The full data dictionary [appears to be provided here](https://leadingage.org/sites/default/files/Skilled%20Nursing%20Facility%20Quality%20Reporting%20Program%20(SNF%20QRP)%20Measures%20on%20Nursing%20Home%20Compare.pdf) and I've included it in the Zenodo upload for this dataset as well, but I had to hunt it down and pull it from a non-government website, so no guarantees.
	
3. `Long-Term_Care_Hospital-General_06-06-2019.csv`
	* Dataset last updated on 6/6/2019
	* Provides a bunch of high-level data about hospitals including street address, ownership type (e.g. non-profit), the total number of patient beds at the facility, and when it was first certified.
	
4. `Long-Term_Care_Hospital-Provider_06-06-2019.csv`
	* Dataset last updated on 6/6/2019
	* Very similar data as provided in the General file, but it doesn't include some of the high-level information (e.g. number of beds) and *does* include some very specific patient-centric scores (e.g. number of patients that experienced serious falls during their stay at the facility)
	* These scores were broken up into their own columns (one column per score type, with its values being the actual scores) to treat them as separate features for modeling. In particular, this was necessary so that they could be scaled as separate features, as each score type seems to be on a different scale (e.g. some are raw counts of patients and some are percentages of patients).
	* The full data dictionary [appears to be provided here](https://data.medicare.gov/views/bg9k-emty/files/34cd7aa0-f28f-4c13-a856-d1a372745aa2) and I've included it in the Zenodo upload for this dataset as well, but I had to hunt it down, so no guarantees.
	
5. `Clinical_Episode-Based_Payment-Hospital_04-28-2019.csv`
	* Dataset last updated on 4/28/2019, but data only cover the calendar year 2017
	* Compares how much Medicare spends on a given medical condition's treatment at specific hospitals to the national average it spends across all hospitals
	* The data are normalized for risk (e.g. by accounting for patients' age and overall health status) and geographic differences in treatment costs
	* These data can be used as a proxy for how reasonable the rates for a given hospital are across procedures and consumables. In other words, these ratios can be used as indicators for the market-competitiveness of chargemaster prices from this hospital when adjusted for an insurance rate (in this case, Medicare). This helps to address the chief complaint of critics of the chargemaster data: that the chargemasters reflect pre-insurance-negotiation rates and that they are nearly random numbers with no statistical meaning that can be derived for patients concerned about what they'll actually pay (as even those paying out of pocket typically receive a discounted rate relative to the chargemaster price). 
	* Including these ratios as features (and also engineering a new feature from their average for a given hospital) will allow for the model to have some concept as to how reasonable the prices from a given hospital may be, **with lower ratios indicating a cheaper price than the Medicare-average and higher ratios indicating a price premium relative to the national Medicare average** (adjusted for geography and risk elements, as mentioned earlier).

6. `Medicare_Spending_Per_Beneficiary-Hospital_Addl_Decimals_04-28-2019.csv`
	* Dataset last updated on 4/28/2019, but data only cover the calendar year 2017
	* Just a general comparison of how a given hospital charges for an average "episode of care" for a patient relative to the national median of hospitals.
	* Higher values = more expensive than the median; lower values = less expensive than the median
	* Like other datasets, this controls for risk and geographic factors affecting prices
	
7. ``