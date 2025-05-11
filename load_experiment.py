from utils import return_data_test, plot_accuracy_over_chunks_2
from main_arch import StandardModel
from self_attention_arch import SelfAttnModel
import torch

#device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#specify model name
folder_name = "standard-model_epoch-2_lr-0.001_batch-64_20250511_182058"
model_name = "trained_model.pth"
#load model
model = StandardModel(228,12)
#model = SelfAttnModel(228,12)

#load state dict into the model
model.load_state_dict(torch.load(f"{folder_name}/{model_name}"))



# Load data
data_raw, data_feature, label_test_oh = return_data_test()

chunk_size = 6000

plot_accuracy_over_chunks_2(model, data_raw, data_feature, label_test_oh, chunk_size, device)

