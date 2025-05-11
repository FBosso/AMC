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
from utils import return_data_train, save_experiment_outputs, return_data_test, return_data_val

#%%

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
    
    
#get training data
data_raw, data_feature, label_train_oh = return_data_train()

#get validation data
data_raw_val, data_feature_val, label_train_oh_val = return_data_val()

# Create Dataset and DataLoader
dataset = RadioDataset(data_feature[:100], data_raw[:100], label_train_oh[:100])
batch_size = 64
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate model
model = StandardModel(228, 12).to(device)

#instantiate potimizer and loss function
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

#%%

# Training loop
epochs = 30
training_loss = []
validation_loss = []

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
        
    
    with torch.inference_mode():
            prediction_val = model(data_feature_val.to(device), data_raw_val.to(device))
            val_loss = criterion(prediction_val.to(device), label_train_oh_val.to(device))
            validation_loss.append(val_loss.item())
            print(validation_loss)
    

    avg_loss = epoch_loss / len(loader)
    training_loss.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} - Avg training Loss: {avg_loss:.4f} - Validation Loss: {val_loss:.4f}")
    
    
#%%
# save model and metrics after training

# Load data
data_raw, data_feature, label_test_oh = return_data_test()
#save experiment
experiment_name = f"standard-model_epoch-{epochs}_lr-{lr}_batch-{batch_size}"
save_experiment_outputs(model, training_loss, validation_loss, data_raw.to(device), data_feature.to(device), label_test_oh.to(device), device, experiment_name=experiment_name, chunk_size=6000)


