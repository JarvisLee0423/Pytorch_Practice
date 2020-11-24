#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/11/24
#   Project Name:       BidirectionalRNNModel.py
#   Description:        Solve the binary sentiment classification problem (Good | Bad) 
#                       by bidirectional LSTM model.
#   Model Description:  Input               ->  Training sentence
#                       Embedding Layer     ->  Converting vocabulary size input into 
#                                               embedding size
#                       B-LSTM Layer        ->  Converting the embedding size input into
#                                               hidden size
#                       Linear Layer        ->  Computing the prediction
#============================================================================================#

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Creating the model.
class BidirectionalRNNModelNN(nn.Module):
    # Creating the constructor.
    def __init__(self, vocabularySize, embeddingSize, hiddenSize, classSize, padIndex):
        # Inheritting the super constructor.
        super(BidirectionalRNNModelNN, self).__init__()
        # Getting the padIndex which is used to pad all the sentences who are not long enough.
        self.padIndex = padIndex
        # Setting the embedding layer.
        self.embedding = nn.Embedding(vocabularySize, embeddingSize)
        # Set the dropout.
        self.dropout = nn.Dropout(p = 0.2)
        # Setting the bidirectional lstm.
        self.lstm = nn.LSTM(embeddingSize, hiddenSize, num_layers = 2, bidirectional = True, batch_first = True, dropout = 0.2)
        # Setting the first full-connected layer.
        self.linear = nn.Linear(2 * hiddenSize, classSize)
    # Defining the forward propagation.
    def forward(self, x, mask):
        # Applying the embedding layer. [batchSize, timeStep] -> [batchSize, timeStep, embeddingSize]
        x = self.embedding(x)
        # Applying the dropout.
        x = self.dropout(x)
        # Getting the sequence length.
        if type(mask) != type([]):
            length = mask.sum(1)
        else:
            length = mask
        # Unpacking the input. [batchSize, timeStep, embeddingSize] -> [batchSize, sentenceLength, embeddingSize]
        x = nn.utils.rnn.pack_padded_sequence(x, length, batch_first = True, enforce_sorted = False)
        # Applying the bidirectional lstm. [batchSize, sentenceLength, embeddingSize] -> [batchSize, sentenceLength, 2 * hiddenSize]
        x, _ = self.lstm(x)
        # Unpadding the input. [batchSize, sentenceLength, 2 * hiddenSize] -> [batchSize, timeStep, 2 * hiddenSize]
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        # Getting the last timeStep's output. [batchSize, timeStep, 2 * hiddenSize] -> [batchSize, 2 * hiddenSize]
        x = x.mean(1).squeeze()
        # Applying the dropout.
        x = self.dropout(x)
        # Applying the linear layer. [batchSize, 2 * hiddenSize] -> [batchSize, 1]
        x = self.linear(x)
        # Returning the result. [batchSize, 1] -> [batchSize]
        return x.squeeze()