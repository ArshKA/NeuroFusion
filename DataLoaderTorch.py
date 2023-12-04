import torch
import glob
import random
import numpy as np


class DatasetTorch():
    'Characterizes a dataset for PyTorch'

    def __init__(self, scan_path, stimuli_path, subject, scan_type, sessions, device='cpu'):
        self.device = device
        # self.scan_files = glob.glob(f'{scan_path}/CSI{subject}_type{scan_type}_sess*.pt')
        # self.stimuli_files = glob.glob(f'{stimuli_path}/CSI{subject}_sess*.pt')
        self.scan_path_gen = lambda x: f'{scan_path}/CSI{subject}_type{scan_type}_sess{x:02}.pt'
        self.stimuli_path_gen = lambda x: f'{stimuli_path}/CSI{subject}_sess{x:02}.pt'
        # self.num_sessions = len(self.scan_files)
        self.sessions = sessions
        self.current_index = -1
    def __len__(self):
        return len(self.sessions)
    def __next__(self):
        if self.current_index == -1 or self.current_index >= len(self.loaded_scan)/37:
            self.load_new_file()

        X = self.loaded_scan[self.index_order[self.current_index]]
        y = self.loaded_stimul[self.index_order[self.current_index]]
        self.current_index += 1

        return X, y

    def load_new_file(self):
        session = random.choice(self.sessions)
        self.loaded_scan = torch.load(self.scan_path_gen(session)).float().to(self.device)
        self.loaded_scan = torch.nan_to_num(self.loaded_scan)
        self.loaded_scan = torch.unsqueeze(torch.swapaxes(self.loaded_scan, -1, 0), 1)
        self.loaded_stimul = torch.load(self.stimuli_path_gen(session)).float().to(self.device)
        self.index_order = np.random.permutation(self.loaded_scan.shape[0]).reshape((-1, 37))
        self.current_index = 0
