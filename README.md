[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![GitHub pull requests](https://img.shields.io/github/issues-pr/suvadeepmaiti/iSLEEPS)
![GitHub issues](https://img.shields.io/github/issues/suvadeepmaiti/iSLEEPS)

# iSLEEPS

This repository has been made to help searcher to have a quick start on analysing the iSLEEPS dataset available here in iHUB-data:
[iSLEEPS dataset](link).

## Introduction
This dataset contains data from externalized DBS patients undergoing simultaneous MEG - STN LFP recordings with (MedOn) and without (MedOn) dopaminergic medication. It has two movement conditions: 1) 5 min of rest followed by static forearm extension (hold) and 2) 5 min of rest followed by self-paced fist-clenching (move). The movement parts contain pauses. Some patients were recorded in resting-state only (rest). The project aimed to understand the neurophysiology of basal ganglia-cortex loops and its modulation by movement and medication.

## Code Structure
├── README.md
├── requirements.txt
├── preprocess
│   ├── __init__.py
│   ├── channel_mapping.py
│   ├── config.py
│   ├── main.py
│   |── numpy_subjects.py
|   |── staging_preprocess.py
├── Technical validations
│   |── 
│   ├── 
├── images
├── demo.ipynb
|── main.py
└── src
