#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/24
#   File Name:      DataPreprocessor.py
#   Description:    This file is used to setting the data preprocessor components.
#============================================================================================#

# Importing the necessary library.
import torch.utils.data as tud
import torchtext
from torchtext.vocab import Vectors

# Creating the training data generator.
class dataLoader():
    # Defining the data generating method.
    @staticmethod
    def Datasets(root, device, vocabularySize, batchSize):
        # Creating the text field which only contains the lower case letters.
        textField = torchtext.data.Field(lower = True)
        # Getting the training, development and testing data.
        trainData, devData, testData = torchtext.datasets.LanguageModelingDataset.splits(path = root, train = 'train.txt', validation = 'dev.txt', test = 'test.txt', text_field = textField)
        # Creating the vocabulary.
        textField.build_vocab(trainData, max_size = vocabularySize)
        # Getting the training, development and testing sets.
        trainSet, devSet, testSet = torchtext.data.BPTTIterator.splits((trainData, devData, testData), batch_size = batchSize, device = device, bptt_len = 10, repeat = False, shuffle = True)
        # Returning the prepared data.
        return textField.vocab, trainSet, devSet, testSet