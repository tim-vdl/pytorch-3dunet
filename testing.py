import h5py
import numpy as np
import matplotlib.pyplot as plt
import time

from multi_slice_viewer.multi_slice_viewer import multi_slice_viewer



file_path = 'data/h5/bad_01_10_15_17.h5'

with h5py.File(file_path, 'r') as input_file:
    print('Input file:')
    print(input_file)
    print()

    print('Input raw file')
    print(input_file['raw'])
    print(np.shape(input_file['raw']))
    multi_slice_viewer(input_file['raw'])

    print('Input label file')
    print(input_file['label'])
    print(np.shape(input_file['label']))

    print(input_file['label'].dtype)

    multi_slice_viewer(input_file['label'])

print()