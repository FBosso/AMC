#%%

from utils import plot_tsne_with_snr_balanced, return_data_train, return_data_test, plot_snr_grouped_confusion_matrices, plot_attention_maps
from models import SelfAttnModel
import torch
from dataset import RadioDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

#%%

#create TSNE plots
data_raw, data_feature, label_train_oh = return_data_train()
plot_tsne_with_snr_balanced(data_raw, data_feature, label_train_oh)

#%%

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Load model
experiment_name = 'selfAttn-model_epoch-100_lr-0.001_batch-64_20250511_220053'
model = SelfAttnModel(228, 12).to(device)    
model.load_state_dict(torch.load(f"models/{experiment_name}/trained_model.pth", map_location=device))
model.eval()

# Load data
raw, feature, y_true = return_data_test()
test_set = RadioDataset(feature, raw, y_true)
batch_size = 64
loader = DataLoader(test_set, batch_size=batch_size)

#%%

# Inference
all_preds = np.zeros(len(test_set), dtype=np.int64)
all_true = np.zeros(len(test_set), dtype=np.int64)

with torch.no_grad():
    idx = 0
    for batch_feat, batch_raw, batch_y in tqdm(loader):
        batch_feat = batch_feat.to(device)
        batch_raw = batch_raw.to(device)
        
        y_hat = model(batch_feat, batch_raw)
        y_pred = torch.argmax(y_hat, dim=1).cpu().numpy()
        
        y_true_indices = torch.argmax(batch_y, dim=1).cpu().numpy()
        
        batch_size_actual = y_pred.shape[0]
        all_preds[idx:idx+batch_size_actual] = y_pred
        all_true[idx:idx+batch_size_actual] = y_true_indices
        idx += batch_size_actual



plot_snr_grouped_confusion_matrices(all_preds,all_true)
    
# %%

plot_attention_maps(feature[-128:],raw[-128:])
# %%
