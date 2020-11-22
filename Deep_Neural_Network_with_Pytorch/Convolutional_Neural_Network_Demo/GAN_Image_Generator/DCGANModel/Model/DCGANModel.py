#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/11/21
#   Project Name:       DCGANModel.py
#   Description:        Build a deconvolutional GAN model to generating the image in 
#                       MNIST datasets.
#   Model Description:  Input               ->  CELEBA Images and Fake Images
#                       Generator           ->  ConvTranspose2d     ->  InChannel:  100
#                                                                   ->  OutChannel: 1024
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     1
#                                                                   ->  Padding:    0
#                                           ->  BatchNorm2d         ->  InChannel:  1024
#                                           ->  ReLu                ->  InPlace:    True
#                                           ->  ConvTranspose2d     ->  InChannel:  1024
#                                                                   ->  OutChannel: 512
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     2
#                                                                   ->  Padding:    1
#                                           ->  BatchNorm2d         ->  InChannel:  512
#                                           ->  ReLu                ->  InPlace:    True
#                                           ->  ConvTranspose2d     ->  InChannel:  512
#                                                                   ->  OutChannel: 256
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     2
#                                                                   ->  Padding:    1
#                                           ->  BatchNorm2d         ->  InChannel:  256
#                                           ->  ReLu                ->  InPlace:    True                                    
#                                           ->  ConvTranspose2d     ->  InChannel:  256
#                                                                   ->  OutChannel: 128
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     2
#                                                                   ->  Padding:    1
#                                           ->  BatchNorm2d         ->  InChannel:  128
#                                           ->  ReLu                ->  InPlace:    True
#                                           ->  ConvTranspose2d     ->  InChannel:  128
#                                                                   ->  OutChannel: 3
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     2
#                                                                   ->  Padding:    1
#                                           ->  Tanh
#                       Discriminator       ->  Conv2d              ->  InChannel:  3
#                                                                   ->  OutChannel: 128
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     2
#                                                                   ->  Padding:    1
#                                           ->  BatchNorm2d         ->  InChannel:  128
#                                           ->  LeakyReLu           ->  Rate:       0.2
#                                                                   ->  InPlace:    True
#                                           ->  Conv2d              ->  InChannel:  128
#                                                                   ->  OutChannel: 256
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     2
#                                                                   ->  Padding:    1
#                                           ->  BatchNorm2d         ->  InChannel:  256
#                                           ->  LeakyReLu           ->  Rate:       0.2
#                                                                   ->  InPlace:    True
#                                           ->  Conv2d              ->  InChannel:  256
#                                                                   ->  OutChannel: 512
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     2
#                                                                   ->  Padding:    1
#                                           ->  BatchNorm2d         ->  InChannel:  512
#                                           ->  LeakyReLu           ->  Rate:       0.2
#                                                                   ->  InPlace:    True
#                                           ->  Conv2d              ->  InChannel:  512
#                                                                   ->  OutChannel: 1024
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     2
#                                                                   ->  Padding:    1
#                                           ->  BatchNorm2d         ->  InChannel:  1024
#                                           ->  LeakyReLu           ->  Rate:       0.2
#                                                                   ->  InPlace:    True
#                                           ->  Conv2d              ->  InChannel:  1024
#                                                                   ->  OutChannel: 1
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     1
#                                                                   ->  Padding:    0
#                                           ->  Sigmoid
#============================================================================================#

# Importing the library.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Creating the model for generator.
class Generator(nn.Module):
    # Creating the constructor.
    def __init__(self, latentSize):
        # Inheritting the super constructor.
        super(Generator, self).__init__()
        # Setting the model.
        self.convTrans2d1 = nn.ConvTranspose2d(latentSize, 1024, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(1024)
        self.convTrans2d2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.convTrans2d3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.convTrans2d4 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.convTrans2d5 = nn.ConvTranspose2d(128, 3, 4, 2, 1)
    # Setting the method to initializing the weight.
    @staticmethod
    def weightInit(model):
        # Getting the model name.
        name = model.__class__.__name__
        # Initializing the weight.
        if name.find("Conv") != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
            nn.init.constant_(model.bias.data, 0)
        elif name.find("BatchNorm") != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
            nn.init.constant_(model.bias.data, 0)
    # Doing the forward propagation.
    def forward(self, x):
        # [batchSize, 100, 1, 1] -> [batchSize, 1024, 4, 4]
        x = self.convTrans2d1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace = True)
        # [batchSize, 1024, 4, 4] -> [batchSize, 512, 8, 8]
        x = self.convTrans2d2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace = True)
        # [batchSize, 512, 8, 8] -> [batchSize, 256, 16, 16]
        x = self.convTrans2d3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace = True)
        # [batchSize, 256, 16, 16] -> [batchSize, 128, 32, 32]
        x = self.convTrans2d4(x)
        x = self.bn4(x)
        x = F.relu(x, inplace = True)
        # [batchSize, 128, 32, 32] -> [batchSize, 3, 64, 64]
        x = self.convTrans2d5(x)
        # Returning the data.
        return torch.tanh(x)

# Creating the model for discrimitor.
class Discrimitor(nn.Module):
    # Creating the constructor.
    def __init__(self):
        # Inheritting the super constructor.
        super(Discrimitor, self).__init__()
        # Setting the model.
        self.conv2d1 = nn.Conv2d(3, 128, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2d2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv2d3 = nn.Conv2d(256, 512, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv2d4 = nn.Conv2d(512, 1024, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.conv2d5 = nn.Conv2d(1024, 1, 4, 1, 0)
    # Doing the forward propagation.
    def forward(self, x):
        # [batchSize, 3, 64, 64] -> [batchSize, 128, 32, 32]
        x = self.conv2d1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2, inplace = True)
        # [batchSize, 128, 32, 32] -> [batchSize, 256, 16, 16]
        x = self.conv2d2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2, inplace = True)
        # [batchSize, 256, 16, 16] -> [batchSize, 512, 8, 8]
        x = self.conv2d3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2, inplace = True)
        # [batchSize, 512, 8, 8] -> [batchSize, 1024, 4, 4]
        x = self.conv2d4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2, inplace = True)
        # [batchSize, 1024, 4, 4] -> [batchSize, 1, 1, 1]
        x = self.conv2d5(x)
        # [batchSize, 1, 1, 1] -> [batchSize, 1]
        x = x.squeeze().unsqueeze(1)
        # Returning the data.
        return torch.sigmoid(x)