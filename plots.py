#%%

from utils import plot_tsne_with_snr, return_data_train, return_data_test, plot_snr_grouped_confusion_matrices
from models import SelfAttnModel
import torch

#%%

#create TSNE plots
data_raw, data_feature, label_train_oh = return_data_train()
#plot_tsne_with_snr(data_raw[-6000:], data_feature[-6000:], label_train_oh[-6000:])

#%%

#create confusion matrices
experiment_name = 'selfAttn-model_epoch-100_lr-0.001_batch-64_20250511_220053'
model = SelfAttnModel(228, 12)    
model.load_state_dict(torch.load(f"models/{experiment_name}/trained_model.pth"))

with model.eval():
    raw, feature, y_true = return_data_test()
    y_hat = model(feature,raw)
    plot_snr_grouped_confusion_matrices(y_true,y_hat)
    
# %%
