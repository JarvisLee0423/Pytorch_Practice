#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/24
#   File Name:      DataPreprocessor.py
#   Description:    This file is used to setting the data preprocessor components.
#============================================================================================#

# Importing the necessary library.
import numpy as np
import random
import torch
import torchtext
import torchtext.vocab as Vectors

# Creating the data generator.
class dataLoader():
    # Defining the data generating method.
    @staticmethod
    def IMDB(root, vocabularySize, batchSize, device):
        # Creating the text field.
        textField = torchtext.data.Field(tokenize = 'spacy', tokenizer_language = 'en_core_web_sm')
        # Creating the label field.
        labelField = torchtext.data.LabelField(dtype = torch.float32)
        # Getting the training and testing data.
        trainData, testData = torchtext.datasets.IMDB.splits(textField, labelField, root = root)
        # Spliting the training data into training data (70%) and development data (30%).
        trainData, devData = trainData.split(random_state = random.seed(1))
        # Creating the vocabulary.
        textField.build_vocab(trainData, max_size = vocabularySize, vectors = 'glove.6B.100d', unk_init = torch.Tensor.normal_, vectors_cache = root + '/glove.6B.100d/')
        labelField.build_vocab(trainData)
        # Getting the train, dev and test sets.
        trainSet, devSet, testSet = torchtext.data.BucketIterator.splits((trainData, devData, testData), batch_size = batchSize, device = device)
        # Returning the training data.
        return textField, trainSet, devSet, testSet