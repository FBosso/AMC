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
    
#create the cnn class
class IQ_cnn(nn.Module):
    def __init__(self):
        #run init of parent module
        super().__init__()
        #define layers
        self.conv1 = nn.Conv2d(1,32,(15,2))
        self.max_pool = nn.MaxPool2d(1,2)
        
        self.depth_conv1 = DepthwiseSeparableConv(32,32)
        
        self.depth_conv2 = DepthwiseSeparableConv(32,64, pooling=True)
        
        self.depth_conv3 = DepthwiseSeparableConv(64,128)
        
        self.depth_conv4 = DepthwiseSeparableConv(128,64, pooling=True)
        
        self.depth_conv5 = DepthwiseSeparableConv(128,128)
        
        self.max_pool2 = nn.MaxPool2d((65,1))
        
        self.fc = nn.Linear(64,64)
        
    def forward(self,x):
        
        x = self.conv1(x) #ok 1
        x = self.max_pool(x) #ok 1
        
        x = self.depth_conv1(x) #no ok
        x = self.depth_conv2(x)
        x = self.depth_conv3(x)
        x = self.depth_conv4(x)
        #x = self.depth_conv5(x)
        x = self.max_pool2(x)
        x = x.squeeze()
        
        return x
        

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, bias=False, pooling=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.pooling = pooling

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 3),
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=(2,1), stride=2)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pooling:
            x = self.pool(x)
        return x