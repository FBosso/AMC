# %%
# import section
import torch
from cnn import IQ_cnn
from mlp import MLP_hand_features
from fusion import Fusion
from torch import optim, nn
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class StandardModel(nn.Module):
    def __init__(self, hand_features_number, n_output_classes):
        super().__init__()

        self.mlp = MLP_hand_features(hand_features_number)
        self.cnn = IQ_cnn()
        self.fusion = Fusion(128)
        self.out = nn.Linear(128, n_output_classes)

    def forward(self, x1, x2):
        hand_out = self.mlp(x1)
        IQ_out = self.cnn(x2)
        cat = torch.cat((hand_out, IQ_out.view(1, 1, 64)), dim=2)
        fusion_out = self.fusion(cat)
        attention = fusion_out * cat
        final_out = self.out(attention)
        return final_out

# instantiate model and move to device
model = StandardModel(228, 12).to(device)

# define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# import data
train_data = h5py.File('HKDD_AMC12/HKDD_AMC12_train.mat')
data_raw = torch.tensor(train_data['XTrainIQ'][:], dtype=torch.float32)
data_feature = torch.tensor(train_data['Feature'][:], dtype=torch.float32).unsqueeze(dim=0)

label_base = np.arange(0, 12)
label_train = label_base.repeat(1000)
label_train = np.tile(label_train, 21)
n_classes = 12
label_train_oh = torch.tensor(np.eye(n_classes)[label_train], dtype=torch.float32).unsqueeze(dim=0)

# move data to device
X1_train = data_feature.to(device)
X2_train = data_raw.to(device)
Y_train = label_train_oh.to(device)

# training loop
epochs = 50
training_loss = []

for epoch in range(epochs):
    model.train()
    predict = model(X1_train, X2_train)
    loss_val = criterion(predict, Y_train)

    training_loss.append(loss_val.item())

    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss_val.item():.4f}")

# plot training loss
plt.figure(figsize=(8, 5))
plt.plot(training_loss, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
