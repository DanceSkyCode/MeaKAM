import os
import pandas as pd
import numpy as np
import h5py
from filter import butter_lowpass_filter

directory = 'D:/dataKOA/predict/data'
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
csv_file = 'D:/dataKOA/predict/measure.csv'
df1 = pd.read_csv(os.path.join(csv_file))
sum = 1
fs = 100.0 
cutoff = 10.0 
def file_sort_key(file_name):
    start_num = int(file_name.split('-')[0])
    return start_num
csv_files = sorted(csv_files, key=file_sort_key)
with h5py.File('dataset.h5', 'w') as hf:
    for i, file in enumerate(csv_files):
        df = pd.read_csv(os.path.join(directory, file))
        input_data = df.iloc[:, 1:31].values
        input_data = butter_lowpass_filter(input_data, cutoff, fs)
        output_data = df.iloc[:, 0].values / 10
        output_data = butter_lowpass_filter(output_data, cutoff, fs)
        measure = df1.iloc[sum-1].values
        dataset_name = os.path.splitext(file)[0] 
        print('\''+dataset_name+'\',')
        hf.create_dataset(f'{dataset_name}_input_data', data=input_data)
        hf.create_dataset(f'{dataset_name}_output_data', data=output_data)
        hf.create_dataset(f'{dataset_name}_measure', data=measure)
        sum = sum + 1
print("Dataset saved to dataset.h5")

with h5py.File('dataset.h5', 'r') as hf:
    dataset_names = list(hf.keys())
    print("Dataset names:", dataset_names)
    first_dataset_input_data = hf[f'{dataset_names[0]}'][:]
    first_dataset_measure_data = hf[f'{dataset_names[1]}'][:]
    first_dataset_output_data = hf[f'{dataset_names[2]}'][:]
    print("First dataset input data:", first_dataset_input_data)
    print("First dataset input data:", first_dataset_input_data.shape)
    print("First dataset measure data:", first_dataset_measure_data)
    print("First dataset measure data:", first_dataset_measure_data.shape)
    print("First dataset output data:", first_dataset_output_data)
    print("First dataset output data:", first_dataset_output_data.shape)