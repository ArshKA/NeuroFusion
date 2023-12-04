import torch
import h5py
import random


class DatasetH5(torch.utils.data.Dataset):
    def __init__(self, scan_path, stimuli_path, subject, scan_type, sessions, device='cpu', norm_path=False):
        self.device = device
        self.sessions = sessions
        self.scans = h5py.File(scan_path, 'r')[f'CSI{subject}/type{scan_type}']
        self.stimuli = h5py.File(stimuli_path, 'r')[f'CSI{subject}']
        self.normalize_scans = lambda x: x
        if norm_path:
            mins = torch.load(f'{norm_path}/min.pt')
            maxs = torch.load(f'{norm_path}/max.pt')
            self.normalize_scans = lambda x: (x-mins)/(maxs-mins)
        self.order = [(s, r) for s in self.sessions for r in range(self.stimuli[f'sess{s:02}'].shape[0])]
        random.shuffle(self.order)

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        ses, run = self.order[idx]
        X = torch.tensor(self.scans[f'sess{ses:02}'][run])
        X = self.normalize_scans(X)
        X = torch.nan_to_num(X)
        X = X * torch.normal(1, .01, size=X.shape)
        X = X.unsqueeze(0)
        X = X.float().to(self.device)
        y = torch.tensor(self.stimuli[f'sess{ses:02}'][run]).float().to(self.device)
        return X, y
