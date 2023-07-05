from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinAct(nn.Module):
    def __init__(self, inplace: bool = False):
        super(SinAct, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return torch.sin(input)

activation_dict = { 'tanh': nn.Tanh(),
                    'relu': nn.ReLU(),
                    'gelu': nn.GELU(),
                    'leaky_relu': nn.LeakyReLU(),
                    'sigmoid': nn.Sigmoid(),
                    'none': nn.Identity(),
                    'split': nn.ModuleList([nn.ReLU(), nn.Identity()]),
                    'sin': SinAct(),
                    }

        

class Flatten(nn.Module): # unnecessary, now there is a Flatten layer natively in torch
    def forward(self, x):
        return x.reshape((x.shape[0], -1))


class DownsampleLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=4, activation='gelu', dropout_rate=0.0, batchnorm=True, bias=True, verbose=False):
        """
        This will do a patched downsampling, where stride=kernel
        """

        super(DownsampleLayer, self).__init__()
        self.dropout_rate=dropout_rate

        stride=kernel
    
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=0, bias=bias)
    
    # Batch norm
        if not batchnorm or batchnorm=='none':
            self.bn = nn.Identity()
        else:
            self.bn   = nn.BatchNorm2d(in_channel)

    # Activation
        self.act = activation_dict.get(activation)
        if self.act is None: 
            print('CnnCell: invalid activation function,\t' +activation+ '\t, using identity (linear)')
            self.act = nn.Tanh()

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.act(x)
       # x = self.bn(x)

        return x

    def named_grad_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params

  
     
class ConvNextCell(nn.Module):
    def __init__(self, in_out_channel, kernel=7, stride=1, dilation=1, activation='gelu', batchnorm=True, dropout_rate=0.0, inv_bottleneck_factor=4, verbose=False, bias=True):
        # Stride and Dilation should always be 1

        super(ConvNextCell, self).__init__()
        self.dropout_rate=dropout_rate
        self.verbose = verbose
        padding=kernel//2


    # Each block will consist of 3 convolutional layers. 
        self.conv0 = nn.Conv2d(in_out_channel, in_out_channel, 
                                groups=in_out_channel, kernel_size=kernel, stride=stride, dilation=dilation, padding=padding, padding_mode='zeros', bias=bias) # used for too-small images
        self.conv1 = nn.Conv2d(in_out_channel, in_out_channel*inv_bottleneck_factor, 
                                kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(in_out_channel*inv_bottleneck_factor, in_out_channel, 
                                kernel_size=1, bias=bias)
        
    # Batch norm
        if not batchnorm or batchnorm=='none':
            self.bn = nn.Identity()
        else:
            self.bn   = nn.BatchNorm2d(in_out_channel)

    # Activation
        self.act = activation_dict.get(activation)
        if self.act is None: 
            print('CnnCell: invalid activation function,\t' +activation+ '\t, using identity (linear)')
            self.act = nn.Tanh()

    def forward(self, x):
        y = self.conv0(x)
        y = self.bn(y)
        y = self.conv1(y)
        y = self.act(y)
        y = self.conv2(y)
        x = y + x
    
        x = nn.Dropout(p=self.dropout_rate)(x)
        return x
            

    def named_grad_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params



