import argparse
import os
import h5py
import numpy as np

def estimate_class_weights(input_path, labels, label_internal_path, ndims):
    """
    Estimate the class weights that should be used for handling class imbalances
    based on the relative abundance of the classes in the dataset.

    :param input_path: path to folder that contains the h5 dataset files
    :param labels: different labels present in the dataset
    :param label_internal_path: internal path in h5 dataset to label array
    :param ndims: specification if label array is 3D or 4D
    """
    file_paths = [os.path.join(input_path, file) for file in os.listdir(input_path)
                  if os.path.isfile(os.path.join(input_path, file)) and file.endswith('.h5')]
    n_labels = len(labels)
    counts   = np.zeros((1, n_labels))

    for file_path in file_paths:
        with h5py.File(file_path, 'r') as file:
            label_array = file[label_internal_path][()].astype(int)
            print(np.unique(label_array))
            for i, lbl in enumerate(labels):
                if ndims == 3:
                    binary_mask = np.where(label_array == lbl, 1, 0)
                    counts[:,i] += np.sum(binary_mask.flatten())
                elif ndims == 4:
                    counts[:,i] += np.sum(label_array[i,:].flatten())
    n_files  = len(file_paths)
    n_voxels = n_files * np.prod(label_array.shape)
    class_weights = 1 - counts / n_voxels
    return class_weights


def main(args):
    labels = [int(lbl) for lbl in args.labels]
    ndims = int(args.ndims)
    class_weights = estimate_class_weights(
        args.input_path,
        labels,
        args.label_internal_path,
        ndims
        )
    print(f'Class weights: {class_weights}')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process user input')
    parser.add_argument('-i', '--input_path', default='./data/h5_3D', type=str,
                        help='Specify path to folder that contains the h5 files'
                        )
    parser.add_argument('--labels', nargs='+', required=True,
                        help='Specify the different labels (classes) that can be present in the label array'
                        )
    parser.add_argument('--label_internal_path', default='label', type=str,
                        help='Specify the internal path in h5 dataset to label array'
                        )
    parser.add_argument('--ndims', type=int, choices=[3,4], required=True,
                        help = 'Specify if h5 datafiles include \
                                3D (DxHxW, every class has a different label) or \
                                4D (CxDxHxW, every class has a different channel with binary mask) arrays'
                        )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

    
    