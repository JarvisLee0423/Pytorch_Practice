#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/11/22
#   Project Name:       HandWrittingNumberModel.py
#   Description:        Build the CNN model to solve the hand writting number problem.
#   Model Description:  Input               ->  Hand writting number Gray-Level Image
#                       Conv2d Layer        ->  InChannel = 1
#                                           ->  OutChannel = 64
#                                           ->  Kernel Size = (3, 3)
#                                           ->  Padding = 1
#                                           ->  Output Size = 28 * 28 * 64
#                       MaxPool2d Layer     ->  Kernel Size = (2, 2)
#                                           ->  Stride = 2
#                                           ->  Output Size = 14 * 14 * 64
#                       Conv2d Layer        ->  InChannel = 64
#                                           ->  OutChannel = 128
#                                           ->  Kernel Size = (5, 5)
#                                           ->  Output Size = 10 * 10 * 128
#                       MaxPool2d Layer     ->  Kernel Size = (2, 2)
#                                           ->  Stride = 2
#                                           ->  Output Size = 5 * 5 * 128
#                       Conv2d Layer        ->  InChannel = 128
#                                           ->  OutChannel = 64
#                                           ->  Kernel Size = (1, 1)
#                                           ->  Output Size = 5 * 5 * 64
#                       Linear Layer        ->  Shrinking the size of the input
#                       Linear Layer        ->  Computing the prediction
#============================================================================================#

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Creating the model.
class HandWrittingNumberModelNN(nn.Module):
    # Creating the constructor.
    def __init__(self, inChannel, classSize):
        # Inheriting the super constructor.
        super(HandWrittingNumberModelNN, self).__init__()
        # Setting the first convolutional layer.
        self.conv1 = nn.Conv2d(inChannel, inChannel, (3, 3), padding = 1)
        # Setting the first point-wise convolutional layer.
        self.pointwise1 = nn.Conv2d(inChannel, 64, (1, 1))
        # Setting the first max pooling layer.
        self.maxPool1 = nn.MaxPool2d((2, 2), 2)
        # Setting the second convolutional layer.
        self.conv2 = nn.Conv2d(64, 64, (5, 5))
        # Setting the second point-wise convolutional layer.
        self.pointwise2 = nn.Conv2d(64, 128, (1, 1))
        # Setting the second max pooling layer.
        self.maxPool2 = nn.MaxPool2d((2, 2), 2)
        # Setting the last convolutional layer.
        self.lastconv = nn.Conv2d(128, 64, (1, 1))
        # Setting the first linear layer.
        self.linear_1 = nn.Linear(1600, 400)
        # Computing the prediction.
        self.linear_2 = nn.Linear(400, classSize)
    # Defining the forward propagation.
    def forward(self, x):
        # Applying the first convolutional layer. [batchSize, 1, 28, 28] -> [batchSize, 1, 28, 28]
        x = self.conv1(x)
        # Applying the first point wise convolutional layer. [batchSize, 1, 28, 28] -> [batchSize, 64, 28, 28]
        x = self.pointwise1(x)
        # Applying the first max pooling layer. [batchSize, 64, 28, 28] -> [batchSize, 64, 14, 14]
        x = self.maxPool1(x)
        # Applying the second convolutional layer. [batchSize, 64, 14, 14] -> [batchSize, 64, 10, 10]
        x = self.conv2(x)
        # Applying the second point wise convolutional layer. [batchSize, 64, 10, 10] -> [batchSize, 128, 10, 10]
        x = self.pointwise2(x)
        # Applying the second max pooling layer. [batchSize, 128, 10, 10] -> [batchSize, 128, 5, 5]
        x = self.maxPool2(x)
        # Applying the last convolutional layer. [batchSize, 128, 5, 5] -> [batchSize, 64, 5, 5]
        x = self.lastconv(x)
        # Flattening the data. [batchSize, 64, 5, 5] -> [batchSize, 1600]
        x = x.view(-1, 1600)
        # Applying the first linear layer. [batchSize, 1600] -> [batchSize, 400]
        x = self.linear_1(x)
        # Applying the second linear layer. [batchSize, 400] -> [batchSize, classSize]
        x = self.linear_2(x)
        # Returning the prediction.
        return x