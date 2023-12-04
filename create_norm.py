import h5py
import numpy as np
import torch

min = np.empty((71, 89, 72))
max = np.empty((71, 89, 72))

with h5py.File('/scratch/arsh/mri_h5/scans/mri_data.hdf5', 'r') as f:
    group = f.get('CSI1/typeD')
    for sess in group.keys():
        data = np.array(group.get(sess)[:])

        sample_min = np.nanmin(data, axis=0)
        sample_min = np.nan_to_num(sample_min, 0)
        min = np.nanmin((min, sample_min), axis=0)

        sample_max = np.nanmax(data, axis=0)
        sample_max = np.nan_to_num(sample_max, 1)
        sample_max[sample_max == 0] = 1
        print((sample_max-sample_min).min())

        max = np.nanmax((max, sample_max), axis=0)



min = torch.tensor(min)
max = torch.tensor(max)

torch.save(min, 'norm/min.pt')
torch.save(max, 'norm/max.pt')
