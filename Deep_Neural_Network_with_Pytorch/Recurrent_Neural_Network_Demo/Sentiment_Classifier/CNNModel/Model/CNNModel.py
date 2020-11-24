#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/11/24
#   Project Name:       CNNModel.py
#   Description:        Solve the binary sentiment classification problem (Good | Bad) 
#                       by CNN model.
#   Model Description:  Input               ->  Training sentence
#                       Embedding Layer     ->  Converting vocabulary size input into 
#                                               embedding size
#                       Conv1d Layer        ->  InChannel = 1
#                                           ->  OutChannel = 70
#                                           ->  Kernel Size = (5, embeddingSize)
#                                           ->  Output Size = (timeStep - 4) * 1 * 70
#                       AvgPool1d Layer     ->  Kernel Size = (2, 1)
#                                           ->  Output Size = (timeStep - 5) * 1 * 70
#                       Conv1d Layer        ->  InChannel = 70
#                                           ->  OutChannel = 100
#                                           ->  Kernel Size = (3, 1)
#                                           ->  Output Size = (timeStep - 7) * 1 * 100
#                       AvgPool1d Layer     ->  Kernel Size = (timeStep - 7, 1)
#                                           ->  Output Size = 1 * 1 * 100
#                       Linear Layer        ->  Computing the prediction
#============================================================================================#

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Creating the model.
class CNNModelNN(nn.Module):
    # Creating the constructor.
    def __init__(self, vocabularySize, embeddingSize, classSize, padIndex):
        # Inheritting the super constructor.
        super(CNNModelNN, self).__init__()
        # Getting the padIndex which is used to pad all the sentences who are not long enough.
        self.padIndex = padIndex
        # Setting the embedding layer.
        self.embedding = nn.Embedding(vocabularySize, embeddingSize)
        # Setting the first convolutional layer.
        self.conv1 = nn.Conv1d(1, 70, (5, embeddingSize))
        # Setting the second convolutional layer.
        self.conv2 = nn.Conv1d(70, 100, (3, 1))
        # Setting the linear layer.
        self.linear = nn.Linear(100, classSize)
    # Defining the forward propagation.
    def forward(self, x):
        # Applying the embedding layer. [timeStep, batchSize] -> [timeStep, batchSize, embeddingSize]
        x = self.embedding(x)
        # Reshaping the input data. [timeStep, batchSize, embeddingSize] -> [batchSize, timeStep, embeddingSize]
        x = x.permute(1, 0, 2)
        # Adding the channel dimension into the input data. [batchSize, timeStep, embeddingSize] -> [batchSize, 1, timeStep, embeddingSize]
        x = x.unsqueeze(1)
        # Applying the first convolutional layer. [batchSize, 1, timeStep, embeddingSize] -> [batchSize, 70, timeStep - 4, 1]
        x = self.conv1(x)
        # Applying the first pooling layer. [batchSize, 70, timeStep - 4, 1] -> [batchSize, 70, timeStep - 5, 1]
        x = F.avg_pool2d(x, (2, 1))
        # Applying the second convolutional layer. [batchSize, 70, timeStep - 5, 1] -> [batchSize, 100, timeStep - 7, 1]
        x = self.conv2(x)
        # Applying the second pooling layer. [batchSize, 100, timeStep - 7, 1] -> [batchSize, 100, 1, 1] -> [batchSize, 100]
        x = F.avg_pool2d(x, (x.shape[2], 1)).squeeze()
        # Applying the linear layer. [batchSize, 100] -> [batchSize, 1]
        x = self.linear(x)
        # Returning the result.
        return x.squeeze()