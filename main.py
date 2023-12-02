import torch

#
# nii_file = nib.load('/scratch/arsh/mri/CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-04.nii.gz')
#
# nii_data = nii_file.get_fdata()


import glob
import nibabel as nib
from tqdm import tqdm
import re

TYPE = 'A'
SUBJECT = 1

# All files and directories ending with .txt and that don't begin with a dot:
all_files = glob.glob(f"/scratch/arsh/mri/CSI{SUBJECT}_GLMbetas-TYPE{TYPE}-*_ses-*.nii.gz")

for name in tqdm(all_files):
    session = re.search(f'CSI{SUBJECT}_GLMbetas-TYPE{TYPE}-ASSUMEHRF_ses-(\d+).nii.gz', name).group(1)
    nii_file = nib.load(name)
    nii_data = nii_file.get_fdata()
    nii_data = torch.tensor(nii_data)

    torch.save(nii_data, f'/scratch/arsh/mri_torch/CSI{SUBJECT}_type{TYPE}_sess{session}.pt')



