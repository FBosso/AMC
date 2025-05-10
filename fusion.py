#import section
from torch import nn

class Fusion(nn.Module):
    def __init__(self,features_in):
        #run init of parent module
        super().__init__()
        #define layers
        self.l1 = nn.Linear(features_in,50)
        self.l2 = nn.Linear(50,50)
        self.l3 = nn.Linear(50,features_in)
        #define activation functions
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
    
    def forward(self,x):
        #do forward pass
        x = self.l1(x)
        x = self.tanh(x)
        x = self.l2(x)
        x = self.tanh(x)
        x = self.l3(x)
        x = self.sigm(x)
        return x