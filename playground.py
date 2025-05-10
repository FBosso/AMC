#import section
from mlp import MLP_hand_features
import numpy as np
import torch
#instantiate mlp
mlp = MLP_hand_features(10)
#create random dataset
data = torch.randn(1,30,10)


from cnn import IQ_cnn

cnn = IQ_cnn()
data_IQ = torch.randn(1,1,128,2)

cnn(data_IQ)
