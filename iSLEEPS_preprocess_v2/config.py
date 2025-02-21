# Path to the raw data directory
# This directory contains the raw sleep (EDF and .xlsx) data files that need to be processed.
raw_file_path = '/scratch/saswatabose/sleep_data/sample'

# Path to the output directory where preprocessed data will be saved
# After processing, the cleaned and transformed data will be stored in this directory.
output_data_path = '/scratch/saswatabose/nimhans_preprocessed_sample/'

# List of modalities to be processed
# Specifies the types of data modalities to process. In this case, we are working with EEG and EOG data.
modality = ['eeg', 'eog']
