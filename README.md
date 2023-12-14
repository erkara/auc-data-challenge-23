# auc-data-challenge-23

The Atlanta University Consortium and Morgan-Stanley organized a data challenge in November-December 2023. The challenge involved identifying promising zip codes for business expansion based on the analysis of 1500 zip codes where the business was already performing well.

I led five amazing Spelman students from the Math Department: Mika Campell, Elon Davis, Nikira A. Walter, Jasmin J. Jean-Louis, and Naomi Logan in this competition. Our team Blue Barbies won the competition by ranking #1.

This repository includes our augmented dataset regarding zip codes. You can read the details in our presentation. We augmented the given dataset, consisting of 1500 US-based zip codes, in a way that we created a unique dataset of its kind that can be super useful for the general public. Here is a brief description of some of the files and folders:

**Blue Barbies Presentation**: this is our winning presentation. I highly suggest to go over this before looking at the codes and datasets.

**data_sources**: this files details the data sources used to augment the initial 1500 zip codes. "final_data.csv" is the final version of the augmented data.
For clarity, here are the columns we added to our data. All states are present and there is no missing data point.

## Dataset Columns Description

| Column Name              | Description                                               |
|--------------------------|-----------------------------------------------------------|
| `zip`                    | Zip Code                                                  |
| `lat`                    | Latitude                                                  |
| `lng`                    | Longitude                                                 |
| `city`                   | City Name                                                 |
| `state_id`               | State Abbreviation                                        |
| `state_name`             | Full State Name                                           |
| `population`             | Population Count                                          |
| `density`                | Population Density (per square km)                        |
| `county_name`            | County Name                                               |
| `target`                 | Target Zip Code(yes if 1)                                 |
| `po_box`                 | PO Box Type (yes if 1)                                    |
| `dist_highway`           | Distance to Nearest Highway (in km)                       |
| `dist2_large_airport`    | Distance to Nearest Large Airport (in km)                 |
| `dist2_medium_airport`   | Distance to Nearest Medium-sized Airport (in km)          |
| `dist_to_shore`          | Distance to Nearest Shoreline (in km)                     |
| `number_of_business`     | Number of Businesses                                      |
| `adjusted_gross_income`  | Adjusted Gross Income in the Area                         |
| `total_income_amount`    | Total Income Amount in the Area                           |
| `number_of_returns`      | Number of Tax Returns Filed                               |

**dataset**: this folder has all the datasets we utulized to create "final_data.csv". 

**data_augment.ipynb**: Python codes used to augment the initial data. Start from the first one and move to the second one.These two files will reproduce the "final_data.csv".

**machine_learning.ipynb**: Python codes for our machine learning approach which includes implementation of One-Class-SVM and Isolation Forest in semi-supervised fashion.
