#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/11/22
#   Project Name:       SimpleGANModel.py
#   Description:        Build a simple GAN model to generating the image in MNIST datastes.
#   Model Description:  Input               ->  MNIST Images and Fake Images
#                       Generator           ->  Linaer (latentSize, 200)
#                                           ->  ReLu
#                                           ->  Linear (200, 400)
#                                           ->  ReLu
#                                           ->  Linear (400, imageSize)
#                                           ->  tanh
#                       Discriminator       ->  Linear (imageSize, 400)
#                                           ->  LeakyReLu (0.2)
#                                           ->  Linear (400, 200)
#                                           ->  LeakyReLu (0.2)
#                                           ->  Linear (200, 1)
#                                           ->  Sigmoid    
#============================================================================================#

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Creating the generator.
class Generator(nn.Module):
    # Creating the constructor.
    def __init__(self, latentSize, imageSize):
        # Inheriting the super constructor.
        super(Generator, self).__init__()
        # Defining the model.
        self.linear1 = nn.Linear(latentSize, 200)
        self.linear2 = nn.Linear(200, 400)
        self.linear3 = nn.Linear(400, imageSize)
    # Defining the forward.
    def forward(self, x):
        # [batchSize, latentSpace] -> [batchSize, 200]
        x = self.linear1(x)
        x = F.relu(x)
        # [batchSize, 200] -> [batchSize, 400]
        x = self.linear2(x)
        x = F.relu(x)
        # [batchSize, 400] -> [batchSize, imageSize]
        x = self.linear3(x)
        # Returning the result.
        return torch.tanh(x)

# Creating the discriminator.
class Discriminator(nn.Module):
    # Creating the constructor.
    def __init__(self, imageSize):
        # Inhertiting the super constructor.
        super(Discriminator, self).__init__()
        # Defining the model.
        self.linear1 = nn.Linear(imageSize, 400)
        self.linear2 = nn.Linear(400, 200)
        self.linear3 = nn.Linear(200, 1)
    # Defining the forward.
    def forward(self, x):
        # [batchSize, latentSpace] -> [batchSize, 400]
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.2)
        # [batchSize, 400] -> [batchSize, 200]
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.2)
        # [batchSize, 200] -> [batchSize, 1]
        x = self.linear3(x)
        # Returning the result.
        return torch.sigmoid(x)