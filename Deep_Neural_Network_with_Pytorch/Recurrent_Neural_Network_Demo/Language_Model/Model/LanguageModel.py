#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/11/24
#   Project Name:       LanguageModel.py
#   Description:        Build the language model by using the pytorch.
#   Model Description:  Input               ->  Training text
#                       Embedding Layer     ->  Converting vocabulary size input into 
#                                               embedding size
#                       LSTM Layer          ->  Converting the embedding size input into
#                                               hidden size
#                       Linear Layer        ->  Converting the hidden size input into
#                                               vocabulary size
#                       Output              ->  The predicted word
#                       Train Target        ->  Predicting the next word according to the
#                                               current word
#============================================================================================#

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Creating the model.
class LanguageModelNN(nn.Module):
    # Creating the constructor.
    def __init__(self, vocabularySize, embeddingSize, hiddenSize):
        # Inheritting the super constructor.
        super(LanguageModelNN, self).__init__()
        # Setting the embedding layer.
        self.embedding = nn.Embedding(vocabularySize, embeddingSize)
        # Setting the LSTM layer.
        self.lstm = nn.LSTM(embeddingSize, hiddenSize)
        # Setting the linear layer.
        self.linear = nn.Linear(hiddenSize, vocabularySize)
    # Defining the forward propagation.
    def forward(self, text, hidden):
        # Encoding the text by embedding layer. [timeStep, batchSize] -> [timeStep, batchSize, embeddingSize]
        text = self.embedding(text)
        # Feeding the data into the LSTM unit. [timeStep, batchSize, embeddingSize] -> [timeStep, batchSize, hiddenSize]
        text, hidden = self.lstm(text, hidden)
        # Reshaping the text to form the data whose second dimension is hiddenSize, in order to letting the linear layer to convert the size of all words into vocabularySize. [timeStep * batchSize, hiddenSize]
        text = text.reshape(-1, text.shape[2])
        # Feeding the data into the linear layer. [timeStep * batchSize, hiddenSize] -> [timeStep * batchSize, vocabularySize]
        text = self.linear(text)
        # Returning the predicted result of the forward propagation.
        return text, hidden
    # Initializing the input hidden.
    def initHidden(self, batchSize, hiddenSize, requireGrad = True):
        # Initializting the weight.
        weight = next(self.parameters())
        # Returning the initialized hidden. [1, batchSize, hiddenSize]: The first dimension represents num_layers * num_directions, num_layers represents how many lstm layers are applied, num_directions represents whether the lstm is bidirectional or not.
        return (weight.new_zeros((1, batchSize, hiddenSize), requires_grad = requireGrad), weight.new_zeros((1, batchSize, hiddenSize), requires_grad = requireGrad))
    # Spliting the historical hidden value from the current training.
    def splitHiddenHistory(self, hidden):
        # Checking whether the input is tensor or not.
        if isinstance(hidden, torch.Tensor):
            return hidden.detach()
        else:
            return tuple(self.splitHiddenHistory(h) for h in hidden)