from torch.utils.data import Dataset

# Custom Dataset
class RadioDataset(Dataset):
    def __init__(self, feature_tensor, raw_tensor, label_tensor):
        self.features = feature_tensor
        self.raw = raw_tensor
        self.labels = label_tensor

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (self.features[idx], self.raw[idx], self.labels[idx])