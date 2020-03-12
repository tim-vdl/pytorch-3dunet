import argparse
import os
import numpy as np
import nibabel as nib
import h5py

def write_h5(file_path, input_data, truth_data, input_name='raw', truth_name='label'):
    with h5py.File(file_path, 'w') as output_file:
            print(f'Writing file: {file_path}')
            output_file.create_dataset(input_name, data = input_data)
            output_file.create_dataset(truth_name, data = truth_data)


def main(args):
    input_path  = args.input_path
    output_path = args.output_path
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    sample_names = [sub for sub in os.listdir(input_path)
                    if os.path.isdir(os.path.join(input_path,sub))]
    sample_paths = [os.path.join(input_path, name) for name in sample_names]

    for sample_name, sample_path in zip(sample_names, sample_paths):
        raw_path    = os.path.join(sample_path, args.raw_name)
        label_path  = os.path.join(sample_path, args.labels_name)

        if not os.path.isfile(raw_path) or not os.path.isfile(label_path):
            continue
        raw     = np.array(nib.load(raw_path).dataobj).astype('double')
        raw     = raw / raw.max() # values in range [0, 1]
        label   = np.array(nib.load(label_path).dataobj).astype('double') - 1
        label[label==-1] = -100 # ignore voxels with value == -1

        present_labels = np.unique(label)
        print(f'Raw array size: {raw.shape}')
        print(f'3D Label array size: {label.shape}')
        print(f'Present labels: {present_labels}')

        output_file_path = os.path.join(output_path, sample_name + '.h5')
        if args.ndims == 4:
            labels = args.labels
            n_labels = len(labels)
            # Create one channel and building block
            label_channel = np.expand_dims(np.zeros_like(label), axis=0)
            # Create 4D array with multiple channels in axis=0 
            labels_4d = np.tile(label_channel, (n_labels, 1, 1, 1))
            # Create binary mask for every label channel
            for i, lbl in enumerate(labels):
                labels_4d[i,] = np.where(label == lbl, 1, 0)
            print(f'4D Label array size {labels_4d.shape}')

            # Create 4D array of raw input data
            # raw_4d = np.tile(np.expand_dims(raw, axis=0), (3, 1, 1, 1))
            # print(f'4D Raw array size {raw_4d.shape}')

            # Save h5 data with 4D labels
            write_h5(output_file_path, raw, labels_4d)
        else:
            # Save h5 data with 3D labels
            write_h5(output_file_path, raw, label)

    print('Done!!!')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process user input')
    parser.add_argument('-i', '--input_path', default='./data/nii', type=str,
                        help='Specify path to folder that contains a subfolder for every \
                              sample in which the nii files are located'
                        )
    parser.add_argument('-o', '--output_path', default='./data/h5', type=str,
                        help='Specify path to folder where output h5 files should be saved'
                        )
    parser.add_argument('--raw_name', default='sample.nii.gz', type=str,
                        help='Specify name of the raw input files'
                        )
    parser.add_argument('--labels_name', default='labels.nii.gz', type=str,
                        help='Specify name of the target labels files'
                        )
    parser.add_argument('--ndims', type=int, choices=[3,4], required=True,
                        help = 'Specify if h5 datafiles should include \
                                3D (DxHxW, every class has a different label) or \
                                4D (CxDxHxW, every class has a different channel with binary mask) arrays'
                        )
    parser.add_argument('--labels', nargs='+', required=True,
                        help='Specify the different labels (classes) that can be present in the label array'
                        )
    args = parser.parse_args()
    args.labels = [int(lbl) for lbl in args.labels]
    args.ndims = int(args.ndims)
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)