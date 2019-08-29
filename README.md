[![DOI](https://zenodo.org/badge/166080583.svg)](https://zenodo.org/badge/latestdoi/166080583)

# Hospital Chargemaster Analysis and Modeling

This is an analysis of publicly-available hospital-level data and (eventually) hospital chargemaster data that uses the Centers for Medicare and Medicaid Services open datasets and the [Dinosaur Dataset](https://vsoch.github.io/datasets/2019/hospital-chargemasters/) as its starting points. 

This project is split into multiple phases:

* **Phase 1:** working with the well-structured and labeled CMS hospital-level data and determining the most interesting feature to predict and how well it can be modeled with those data alone
* **Phase 1.5:** after doing analysis in Phase 1, it became evident that more datasets beyond chargemasters were necessary (e.g. geolocations of major cities). This phase will focus on adding in those datasets and measuring improvement (if any) of the models for the chosen target variables.
* **Phase 2:** adding in the hospital chargemaster data as new features (e.g. the ratio of a given hospital's cost for a given procedure relative to the average or median of all the hospitals for which we have data).

### Table of Contents

1. [The Problem](#problem_statement)
2. [The Data](#data)
	1. [CMS-Provided Hospital Data](#hospital_data)
	2. [Chargemaster Data Collection Process](#charge_data)
3. [The Analyses](#analysis)
4. [Related Media](#articles)


## So what are you trying to achieve here?

The US healthcare industry is disturbingly opaque in its pricing and quality of service. It seems to be a market that is not directed by typical economic forces due to a large information asymmetry between the patient and healthcare provider (ever tried to ask your insurance company to commit to how much you'd have to pay for a procedure that isn't 100% covered?). Additionally, there are often significant stresses on the consumer (AKA patient) at the point of purchase, making the decisions less economic in nature and more emotional.

This project and its corresponding repository is intended to try and predict patient quality of care metrics that are relevant in decided the overall quality of hospital from a patient's perspective (e.g. things like how many deaths occur there every year due to serious post-surgical complications). The hope is that patients could use these results to make more informed decisions about where to seek their healthcare, and that the entire US healthcare industry could be made just a little bit more accountable.

## What are these data? <a name="data"></a>

### CMS-Provided Hospital Data <a name="hospital_data"></a>

In order to provide enough features/variables at the hospital level to build predictive models, hospital-level datasets are required that cover a wide range of variables. Metadata and information about the source tables and columns can be found in [the metadata file](metadata.csv) and [the data dictionary](Medicare_Hospitals_DataDictionary.pdf), resp. All of these datasets were accessed programmatically using [the data.medicare.gov APIs for the Hospital Compare datasets](https://data.medicare.gov/data/hospital-compare). Even more information can be found by exploring the [Phase I data wrangling Jupyter notebook](PhaseI_CMS_Data_Engineering.ipynb), regarding decisions and transformations applied. One common element you'll find with all of these data is that they largely relate to inpatient hospital stays for acute care (short-term treatment, instead of long-term situations like nursing homes).

Here I provide some brief descriptions for the versioned datasets used in this work:

1. Hospital General Information
	* A list of all hospitals that have been registered with Medicare. The list includes addresses, phone numbers, hospital type, and overall hospital rating.
3. Complications and Deaths - Hospital
	* Complications and deaths data provided by the hospitals themselves. These data include the hip/knee complication measure, the CMS Patient Safety Indicators, and 30-day death rates.
4. Healthcare Associated Infections - Hospital
	* Information on infections that occur while the patients are in the hospital
5. Unplanned Hospital Visits - Hospital
	* Hospital-reported data for the hospital return days (or excess days in acute care [EDAC]), indicating amounts of time required for patients to return to a hospital as a result of readmission for a previously-treated issue
6. Hospital Readmissions Reduction Program
	* This extends the Unplanned Hospital Visits data by normalizing readmission rates to a statisically-derived similar average-performing hospital (higher values meaning the hospital in question has more readmissions in a 30-day period than is expected nationally). 
7. Medicare Spending Per Beneficiary – Hospital Additional Decimal Places
	* Shows whether Medicare spends more, less, or about the same for an episode of care at a specific hospital compared to all hospitals nationally
8. Outpatient Imaging Efficiency - Hospital
	* Hospital-provided data about the use of medical imaging in their facilities for outpatients
9. Patient survey (HCAHPS) - Hospital
	* Response date from the HCAHPS national, standardized survey of hospital patients about their experiences during a recent inpatient hospital stay
10. Structural Measures - Hospital
	* A list of hospitals and the structural measures they report (e.g. if they utilize electronic health records systems, AKA EHRs)

### Hospital Chargemaster Data <a name="charge_data"></a>

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


**How do the chargemaster data get gathered?**

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


## So...what are the results? <a name="analysis"></a>

A lot of work has gone into the analysis thus far (and it is ongoing), so check the [analysis notebook](Analysis.ipynb) for updates.


## Can I See the Results Without Looking Through Your Code? <a name="articles"></a>

Sure! Here's what I've written up about this analysis and these data in a way that skips all of the details and gets to the juiciest bits:

1. [*Does Where You Live Affect Your Quality of Hospital Care?*](https://medium.com/swlh/does-where-you-live-affect-your-quality-of-hospital-care-9acaf59a9f99)