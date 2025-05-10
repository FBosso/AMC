#%%
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import h5py
import matplotlib.pyplot as plt

from cnn import IQ_cnn
from mlp import MLP_hand_features
from fusion import Fusion

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model definition
class StandardModel(nn.Module):
    def __init__(self, hand_features_number, n_output_classes):
        super().__init__()
        self.mlp = MLP_hand_features(hand_features_number)  # [B, 1, 64]
        self.cnn = IQ_cnn()                                 # [B, 256] or [B, C, H, W]
        self.fusion = Fusion(320)                           # 64 (MLP) + 256 (CNN)
        self.out = nn.Linear(320, n_output_classes)

    def forward(self, x1, x2):
        hand_out = self.mlp(x1)                 # [B, 1, 64]
        IQ_out = self.cnn(x2)                   # could be [B, 256] or [B, C, H, W]

        if IQ_out.dim() == 2:
            IQ_out = IQ_out.unsqueeze(1)        # -> [B, 1, 256]
        elif IQ_out.dim() == 4:
            IQ_out = torch.flatten(IQ_out, start_dim=1)  # [B, -1]
            IQ_out = IQ_out[:, :256]            # Trim/pad to 256-dim if needed
            IQ_out = IQ_out.unsqueeze(1)        # -> [B, 1, 256]

        cat = torch.cat((hand_out, IQ_out), dim=2)  # [B, 1, 320]
        fusion_out = self.fusion(cat)
        attention = fusion_out * cat
        final_out = self.out(attention)
        return final_out.squeeze(1)             # -> [B, n_classes]

# Load data
train_data = h5py.File('HKDD_AMC12/HKDD_AMC12_train.mat')

# IQ data: [252000, 1, 512, 2]
X2_train = torch.tensor(train_data['XTrainIQ'][:], dtype=torch.float32)

# Handcrafted features: [1, 252000, 228] -> [252000, 1, 228]
X1_raw = torch.tensor(train_data['Feature'][:], dtype=torch.float32)
X1_train = X1_raw.squeeze(0).unsqueeze(1)

# Labels
label_base = np.arange(0, 12)
label_train = label_base.repeat(1000)
label_train = np.tile(label_train, 21)  # -> 252000
label_train = torch.tensor(label_train, dtype=torch.long)

# Dataset and DataLoader
batch_size = 64
dataset = TensorDataset(X1_train, X2_train, label_train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, loss, optimizer
model = StandardModel(hand_features_number=228, n_output_classes=12).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 50
training_loss = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for x1_batch, x2_batch, y_batch in train_loader:
        x1_batch = x1_batch.to(device)
        x2_batch = x2_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        output = model(x1_batch, x2_batch)
        loss_val = criterion(output, y_batch)
        loss_val.backward()
        optimizer.step()

        epoch_loss += loss_val.item()

    avg_loss = epoch_loss / len(train_loader)
    training_loss.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# Plot training loss
plt.plot(training_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
