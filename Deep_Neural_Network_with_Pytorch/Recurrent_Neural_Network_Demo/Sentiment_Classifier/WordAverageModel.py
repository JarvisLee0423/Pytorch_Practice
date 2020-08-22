#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/08/22
#   Project Name:       WordAverageModel.py
#   Description:        Solve the sentiment classification problem by word average model.
#   Model Description:  Input               ->  Training sentence
#                       Embedding Layer     ->  Converting vocabulary size input into 
#                                               embedding size
#                       Average Pooling     ->  Computing the encode for each sentence
#                       Linear Layer        ->  Converting the embedding size input into
#                                               hidden size
#                       ReLu                ->  Activation Function
#                       Linear Layer        ->  Computing the prediction.
#============================================================================================#

# Importing the necessary library.
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tud
import torchtext
import torchtext.vocab as Vectors

# Fixing the computer device and random seed.
# Indicating whether the computer has the GPU or not.
if torch.cuda.is_available:
    # Fixing the random seed.
    torch.cuda.manual_seed(1)
    # Fixing the computer device.
    torch.cuda.set_device(0)
    # Setting the computer device.
    device = 'cuda'
else:
    # Fixing the random seed.
    torch.cuda.manual_seed(1)
    # Setting the computer device.
    device = 'cpu'

# Setting the hyperparameters.
# The value of the vocabulary size.
vocabularySize = 10000
# The value of the embedding size.
embeddingSize = 100
# The value of the hidden size.
hiddenSize = 100
# The number of classes.
classSize = 1
# The value of the batch size.
batchSize = 128
# The value of the epoch.
epoches = 10

# Creating the data generator.
class dataGenerator():
    # Defining the data generating method.
    @staticmethod
    def generator(vocabulaySize, batchSize):
        # Creating the text field.
        textField = torchtext.data.Field(tokenize = 'spacy', tokenizer_language = 'en_core_web_sm')
        # Creating the label field.
        labelField = torchtext.data.LabelField(dtype = torch.float32)
        # Getting the training and testing data.
        trainData, testData = torchtext.datasets.IMDB.splits(textField, labelField, root = './Datasets/IMDB/')
        # Spliting the training data into training data (70%) and development data (30%).
        trainData, devData = trainData.split(random_state = random.seed(1))
        # Creating the vocabulary.
        textField.build_vocab(trainData, max_size = vocabularySize, vectors = 'glove.6B.100d', unk_init = torch.Tensor.normal_, vectors_cache = './Datasets/IMDB/')
        labelField.build_vocab(trainData)
        # Getting the train, dev and test sets.
        trainSet, devSet, testSet = torchtext.data.BucketIterator.splits((trainData, devData, testData), batch_size = batchSize, device = device)
        # Returning the training data.
        return textField, trainSet, devSet, testSet

# Training the model.
if __name__ == "__main__":
    #pass
    textField, trainSet, devSet, testSet = dataGenerator.generator(vocabularySize, batchSize)
    print(textField)
    print(trainSet)
    print(devSet)
    print(testSet)