# %%
import torch
from torch.utils.data import Dataset, DataLoader
from cnn import IQ_cnn
from mlp import MLP_hand_features
from fusion import Fusion
from torch import optim, nn
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Model definition
class StandardModel(nn.Module):
    def __init__(self, hand_features_number, n_output_classes):
        super().__init__()
        self.mlp = MLP_hand_features(hand_features_number)
        self.cnn = IQ_cnn()
        self.fusion = Fusion(128)
        self.out = nn.Linear(128, n_output_classes)

    def forward(self, x1, x2):
        # x1: (B, 1, F), x2: (B, 1, 128, 2)
        hand_out = self.mlp(x1)  # (B, 1, 64)
        iq_out = self.cnn(x2)    # (B, 1, 64)
        cat = torch.cat((hand_out, iq_out), dim=1)  # (B, 1, 128)
        fusion_out = self.fusion(cat)               # (B, 1, 128)
        attention = fusion_out * cat                # (B, 1, 128)
        final_out = self.out(attention).squeeze(1)  # (B, n_classes)
        return final_out

# Load data
train_data = h5py.File('HKDD_AMC12/HKDD_AMC12_train.mat')
data_raw = torch.tensor(train_data['XTrainIQ'][:], dtype=torch.float32)  # shape: (N, 1, 128, 2)
data_feature = torch.tensor(train_data['Feature'][:], dtype=torch.float32)  # shape: (N, 1, 228)

label_base = np.arange(0, 12)
label_train = label_base.repeat(1000)
label_train = np.tile(label_train, 21)
n_classes = 12
label_train_oh = torch.tensor(np.eye(n_classes)[label_train], dtype=torch.float32)

# Create Dataset and DataLoader
dataset = RadioDataset(data_feature, data_raw, label_train_oh)
batch_size = 64
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate model
model = StandardModel(228, n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 50
training_loss = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for batch_X1, batch_X2, batch_Y in tqdm(loader):
        batch_X1 = batch_X1.to(device)
        batch_X2 = batch_X2.to(device)
        batch_Y = batch_Y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_X1, batch_X2)
        loss_val = criterion(predictions, batch_Y)
        loss_val.backward()
        optimizer.step()

        epoch_loss += loss_val.item()

    avg_loss = epoch_loss / len(loader)
    training_loss.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(training_loss, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
