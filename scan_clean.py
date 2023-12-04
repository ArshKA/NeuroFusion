import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm
import re
import torch
import h5py

TYPE = 'D'
SUBJECT = 1

# All files and directories ending with .txt and that don't begin with a dot:
all_files = glob.glob(f"/scratch/arsh/raw/scans/CSI{SUBJECT}_*-TYPE{TYPE}-*_ses-*.nii.gz")
print(all_files)

with h5py.File('/scratch/arsh/mri_h5/scans/mri_data.hdf5', 'a') as file:
    group = file.require_group(f"CSI{SUBJECT}/type{TYPE}")
    for name in tqdm(all_files):
        session = re.search(f'CSI{SUBJECT}_GLMbetas-TYPE{TYPE}-.+_ses-(\d+).nii.gz', name).group(1)
        nii_file = nib.load(name)
        nii_data = nii_file.get_fdata()
        nii_data = np.moveaxis(nii_data, -1, 0)

        # torch.save(nii_data, f'/scratch/arsh/mri_torch/scans/CSI{SUBJECT}_type{TYPE}_sess{session}.pt')

        group.create_dataset(f"sess{session}", data=nii_data)

