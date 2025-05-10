#import section
from torch import nn

class MLP_hand_features(nn.Module):
    def __init__(self,features_in):
        #run init of parent module
        super().__init__()
        #define layers
        self.l1 = nn.Linear(features_in,128)
        self.l2 = nn.Linear(128,96)
        self.l3 = nn.Linear(96,64)
        #define activation function
        self.act = nn.ReLU()
    
    def forward(self,x):
        #do forward pass
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        x = self.act(x)
        x = self.l3(x)
        x = self.act(x)
        
        return x