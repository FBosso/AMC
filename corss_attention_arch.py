# %%
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import RadioDataset
from fusion import FusionSelfAttn
from torch import optim, nn
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import return_data_train, save_experiment_outputs, return_data_test, return_data_val
from models import CrossAttnModel

#%%

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
    
    
#get training data
data_raw, data_feature, label_train_oh = return_data_train()
#get validation data
data_raw_val, data_feature_val, label_train_oh_val = return_data_val()

# Create Dataset and DataLoader
batch_size = 64
dataset = RadioDataset(data_feature, data_raw, label_train_oh)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate model
model = CrossAttnModel(228, 12).to(device)

#instantiate potimizer and loss function
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

#%%

# Training loop
epochs = 100
training_loss = []
validation_loss = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    #iterate over batches
    for batch_X1, batch_X2, batch_Y in tqdm(loader):
        batch_X1 = batch_X1.to(device)
        batch_X2 = batch_X2.to(device)
        batch_Y = batch_Y.to(device)
        #put gradient to zero
        optimizer.zero_grad()
        #forward pass
        predictions = model(batch_X1, batch_X2)
        loss_val = criterion(predictions, batch_Y)
        #backpropagate
        loss_val.backward()
        #optimize
        optimizer.step()
        #keep track of the training loss
        epoch_loss += loss_val.item()
        
    #keep track of the validation loss
    with torch.inference_mode():
            prediction_val = model(data_feature_val.to(device), data_raw_val.to(device))
            val_loss = criterion(prediction_val.to(device), label_train_oh_val.to(device))
            validation_loss.append(val_loss.item())
            
    #compute the average of the training loss across batches
    avg_loss = epoch_loss / len(loader)
    training_loss.append(avg_loss)
    #print the losses (train an val)
    print(f"Epoch {epoch+1}/{epochs} - Avg training Loss: {avg_loss:.4f} - Validation Loss: {val_loss:.4f}")
    
    
#%%
# save model and metrics after training

# Load data
data_raw, data_feature, label_test_oh = return_data_test()
#save experiment
experiment_name = f"crossAttn-model_epoch-{epochs}_lr-{lr}_batch-{batch_size}"
save_experiment_outputs(model, training_loss, validation_loss, data_raw.to(device), data_feature.to(device), label_test_oh.to(device), device, experiment_name=experiment_name, chunk_size=6000)
