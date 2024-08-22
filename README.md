[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![GitHub pull requests](https://img.shields.io/github/issues-pr/suvadeepmaiti/iSLEEPS)
![GitHub issues](https://img.shields.io/github/issues/suvadeepmaiti/iSLEEPS)

# iSLEEPS

This repository has been made to help searcher to have a quick start on analysing the iSLEEPS dataset available here in iHUB-data:
[iSLEEPS dataset](link).

## Introduction
**1. Dataset**: The dataset consists of polysomnography (PSG) patient recordings during sleep studies of ischemic stroke patients, capturing brainwave activity across different sleep stages. It includes multiple channels of raw PSG data, annotated with sleep stages and additional annotations such as sleep apnea events, providing a comprehensive resource for sleep research and machine learning model development.

**2. Purpose:** The purpose of the preprocessing code is to take the raw EDF data and corresponding annotation .xlsx files, match the 30-second epochs, and transform them into .npz format suitable for model training. Additionally, it provides baseline scripts to help users get started with training machine learning models on the processed data.

## Repository Structure

- **preprocess**: Script to preprocess raw data, matching 30-second epochs and converting them into .npz format for model training.
- **explore.ipynb**: Basic exploratory data analysis (EDA) script to help understand the dataset.
- **models**: Contains baseline model training scripts to get started with training deep learning models on the processed data.
- **utils/**: Directory containing utility functions for data loading and processing.
- **requirements.txt**: List of Python dependencies needed to run the scripts.





## Code Structure

```

├── README.md
├── requirements.txt
├── preprocess
│   ├── __init__.py
│   ├── channel_mapping.py
│   ├── config.py
│   ├── main.py
│   |── numpy_subjects.py
│   |── staging_preprocess.py
├── Technical validations
│   |── proposal.pdf
│   ├── final-eval.pdf
├── images
├── demo.ipynb
|── main.py
└── src

```

