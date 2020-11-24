#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/11/24
#   Project Name:       WordAverageModel.py
#   Description:        Solve the binary sentiment classification problem (Good | Bad) 
#                       by word average model.
#   Model Description:  Input               ->  Training sentence
#                       Embedding Layer     ->  Converting vocabulary size input into 
#                                               embedding size
#                       Average Pooling     ->  Computing the encode for each sentence
#                       Linear Layer        ->  Converting the embedding size input into
#                                               hidden size
#                       ReLu                ->  Activation Function
#                       Linear Layer        ->  Computing the prediction
#============================================================================================#

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Creating the model.
class WordAverageModelNN(nn.Module):
    # Creating the constructor.
    def __init__(self, vocabularySize, embeddingSize, hiddenSize, classSize, padIndex):
        # Inheritting the super constructor.
        super(WordAverageModelNN, self).__init__()
        # Getting the padIndex which is used to pad all the sentences who are not long enough.
        self.padIndex = padIndex
        # Setting the embedding layer.
        self.embedding = nn.Embedding(vocabularySize, embeddingSize)
        # Setting the hidden layer.
        self.linear_1 = nn.Linear(embeddingSize, hiddenSize)
        # Setting the classifier.
        self.linear_2 = nn.Linear(hiddenSize, classSize)
    # Defining the forward propagation.
    def forward(self, x):
        # Applying the embedding layer. [timeStep, batchSize] -> [timeStep, batchSize, embeddingSize]
        x = self.embedding(x)
        # Reshaping the x which invers the first and second dimensions of the input x. [timeStep, batchSize, embeddingSize] -> [batchSize, timeStep, embeddingSize]
        x = x.permute(1, 0, 2)
        # Applying the average pooling to compute the encode of the input sentence. [batchSize, timeStep, embeddingSize] -> [batchSize, 1, embeddingSize]
        x = F.avg_pool2d(x, (x.shape[1], 1))
        # Squeezing the data. [batchSize, 1, embeddingSize] -> [batchSize, embeddingSize]
        x = x.squeeze(1)
        # Applying the first full-connected layer. [batchSize, embedding] -> [batchSize, hiddenSize]
        x = self.linear_1(x)
        # Applying the ReLu activation function.
        x = F.relu(x)
        # Applying the classifier. [batchSize, hiddenSize] -> [batchSize, classSize]
        x = self.linear_2(x)
        # Returning the prediction. [batchSize, classSize] -> [batchSize]
        return x.squeeze()