#import section
import torch
from torch import nn
from pieces import MLP_hand_features, IQ_cnn
from fusion import Fusion, FusionSelfAttn, FusionCrossAttn

# Model definition
class StandardModel(nn.Module):
    def __init__(self, hand_features_number, n_output_classes):
        super().__init__()
        self.mlp = MLP_hand_features(hand_features_number)
        self.cnn = IQ_cnn()
        self.fusion = Fusion(128)
        self.out = nn.Linear(128, n_output_classes)

    def forward(self, x1, x2, return_attention=False):
        # x1: (B, 1, F), x2: (B, 1, 128, 2)
        hand_out = self.mlp(x1)  # (B, 1, 64)
        iq_out = self.cnn(x2)    # (B, 1, 64)
        cat = torch.cat((hand_out, iq_out), dim=1)  # (B, 1, 128)
        fusion_out = self.fusion(cat)               # (B, 1, 128)
        attention = fusion_out * cat                # (B, 1, 128)
        final_out = self.out(attention).squeeze(1)  # (B, n_classes)
        
        if return_attention == True:
            return fusion_out
        
        return final_out
    

# Model definition
class SelfAttnModel(nn.Module):
    def __init__(self, hand_features_number, n_output_classes):
        super().__init__()
        self.mlp = MLP_hand_features(hand_features_number)
        self.cnn = IQ_cnn()
        self.fusion = FusionSelfAttn(num_classes=n_output_classes)
        

    def forward(self, x1, x2, return_attention=False):
        # x1: (B, 1, F), x2: (B, 1, 128, 2)
        hand_out = self.mlp(x1)  # (B, 1, 64)
        iq_out = self.cnn(x2)    # (B, 1, 64)
        cat = torch.cat((hand_out, iq_out), dim=1)  # (B, 1, 128)
        fusion_out = self.fusion(cat, return_attention=return_attention)               # (B, 1, 128)
        
        return fusion_out
    
    
# Model definition
class CrossAttnModel(nn.Module):
    def __init__(self, hand_features_number, n_output_classes):
        super().__init__()
        self.mlp = MLP_hand_features(hand_features_number)
        self.cnn = IQ_cnn()
        self.fusion = FusionCrossAttn(num_classes=n_output_classes)
        

    def forward(self, x1, x2, return_attention=False):
        # x1: (B, 1, F), x2: (B, 1, 128, 2)
        hand_out = self.mlp(x1)  # (B, 1, 64)
        iq_out = self.cnn(x2)    # (B, 1, 64)
        cat = torch.cat((hand_out, iq_out), dim=1)  # (B, 1, 128)
        fusion_out = self.fusion(cat, return_attention=return_attention)               # (B, 1, 128)
        
        return fusion_out