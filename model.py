import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
                                            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7),
                                            nn.MaxPool2d(kernel_size=2, stride=2),
                                            nn.ReLU(True),
                                            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5),
                                            nn.MaxPool2d(kernel_size=2, stride=2),
                                            nn.ReLU(True)
                                        )
        self.regression_layer = nn.Sequential(
                                                nn.Linear(10 * 3 * 3, 32),
                                                nn.ReLU(True),
                                                nn.Linear(32, 3 * 2)
                                            )   

    """
        Affine Matrix Aθ

        Aθ = [ 
                [θ11, θ12, θ13],
                [θ21, θ22, θ23]
            ]

    """
    def localization_net(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3) # reshape for 10 * 3 * 3 columns
        theta = self.regression_layer(xs)
        theta = theta.view(-1, 2, 3) # reshape for 2 columns and 3 channels

        return theta

    # https://pytorch.org/docs/0.3.1/nn.html#torch.nn.functional.affine_grid
    def grid_generator(self, theta, size):
        return F.affine_grid(theta, size)

    # https://pytorch.org/docs/0.3.1/nn.html#torch.nn.functional.grid_sample
    def sampler(self, x, grid):
        return F.grid_sample(x, grid)

    # Spatial transformer network forward function
    def forward(self, x):
        theta = self.localization_net(x)
        grid  = self.grid_generator(theta, x.size()) 
        x     = self.sampler(x, grid)

        return x

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        # Spatial Transformer Network
        self.stn = STN()

        # Rest of the Neural Network
        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.drop = nn.Dropout2d()
        self.fully_con1 = nn.Linear(320, 50)
        self.fully_con2 = nn.Linear(50, 10)

    def forward(self, x):
        # Spatial Transfomer Network
        x = self.stn.forward(x)

        # Perform forward pass
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        
        x = x.view(-1, 320)
        x = self.fully_con1(x)
        x = F.relu(x)

        x = F.dropout(x, training=self.training)
        x = self.fully_con2(x)

        return F.log_softmax(x, dim=1)