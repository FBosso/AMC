#import section
import h5py
import torch
import numpy as np
from dataset import RadioDataset
from torch.utils.data import DataLoader


def return_dataloader(data_path, batch_size):
    # Load data
    train_data = h5py.File(data_path)
    data_raw = torch.tensor(train_data['XTrainIQ'][:], dtype=torch.float32)  # shape: (N, 1, 128, 2)
    data_feature = torch.tensor(train_data['Feature'][:], dtype=torch.float32)  # shape: (N, 1, 228)

    label_base = np.arange(0, 12)
    label = label_base.repeat(1000)
    label = np.tile(label, 21)
    n_classes = 12
    label_oh = torch.tensor(np.eye(n_classes)[label], dtype=torch.float32)

    # Create Dataset and DataLoader
    dataset = RadioDataset(data_feature, data_raw, label_oh)
    batch_size = 64
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader