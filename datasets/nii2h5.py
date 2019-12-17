import os
import numpy as np
import nibabel as nib
import h5py

def main():
    input_path = 'data/nii'
    output_path = 'data/h5'

    sample_names = [sub for sub in os.listdir(input_path)
                    if os.path.isdir(os.path.join(input_path,sub))]
    sample_paths = [os.path.join(input_path, name) for name in sample_names]

    for sample_name, sample_path in zip(sample_names, sample_paths):
        raw_path = os.path.join(sample_path, 'sample.nii.gz')
        label_path = os.path.join(sample_path, 'labels.nii.gz')
        raw = np.array(nib.load(raw_path).dataobj)
        raw = raw.astype('double')
        raw = raw / raw.max()
        label = np.array(nib.load(label_path).dataobj)
        label = label.astype('double')
        print(np.unique(label))

        print(f'Raw image size: {raw.shape}')
        print(f'Label image size {label.shape}')

        output_file_path = os.path.join(output_path, sample_name + '.h5')
        with h5py.File(output_file_path, 'w') as output_file:
            print(f'Writing file: {output_file_path}')
            output_file.create_dataset('raw', data = raw)
            output_file.create_dataset('label', data = label)

    print('Done!!!')

if __name__ == '__main__':
    main()