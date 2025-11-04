import h5py
import numpy as np
from config import SUBJECTS
def load_dataset(h5_file):
    input_data_list = []
    output_data_list = []
    measure_data_list = []
    with h5py.File(h5_file, 'r') as hf:
        for subject in SUBJECTS:
            if subject + '_input_data' in hf and subject + '_output_data' in hf and subject + '_measure' in hf:
                input_data = hf[subject + '_input_data'][:][np.newaxis, ...]
                output_data = hf[subject + '_output_data'][:][np.newaxis, ...]
                measure_data = hf[subject + '_measure'][:][np.newaxis, ...]
                # input_data = (input_data - np.min(input_data, axis=1, keepdims=True)) / (np.max(input_data, axis=1, keepdims=True) - np.min(input_data, axis=1, keepdims=True))
                # output_data = (output_data - np.min(output_data, axis=1, keepdims=True)) / (np.max(output_data, axis=1, keepdims=True) - np.min(output_data, axis=1, keepdims=True))
                # measure_data = (measure_data - np.min(measure_data, axis=1, keepdims=True)) / (np.max(measure_data, axis=1, keepdims=True) - np.min(measure_data, axis=1, keepdims=True))
                
                input_data_list.append(input_data)
                output_data_list.append(output_data)
                measure_data_list.append(measure_data)
            else:
                print(f"Data for subject {subject} not found in the HDF5 file.")
    input_data = np.concatenate(input_data_list, axis=0)
    output_data = np.concatenate(output_data_list, axis=0)
    measure_data = np.concatenate(measure_data_list, axis=0)
    
    return input_data, output_data, measure_data

