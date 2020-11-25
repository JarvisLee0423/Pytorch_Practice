#============================================================================================#
#   Copyright:          JarvisLee
#   Date:               2020/11/25
#   File Name:          DataPreprocessor.py
#   Description:        This file is used to setting the data preprocessor components.
#============================================================================================#

# Importing the necessary library.
import numpy as np
import torch
import torch.utils.data as tud
from collections import Counter

# Creating the training data generator.
class dataLoader(tud.Dataset):
    # Creating the constructor.
    def __init__(self, text, itos, stoi, wordFreq, contextSize, negativeSampling):
        # Inheritting the super class.
        super(dataLoader, self).__init__()
        # Getting the context size.
        self.contextSize = contextSize
        # Getting the negative sampling.
        self.negativeSampling = negativeSampling
        # Getting the training text.
        self.text = text
        # Getting the itos tool.
        self.itos = itos
        # Getting the stoi tool.
        self.stoi = stoi
        # Getting the sampling distribution.
        self.wordFreq = wordFreq
        # Encoding the words in the text as one-hot encodes.
        self.textEncode = torch.LongTensor([self.stoi.get(word, self.stoi.get('<unk>')) for word in self.text])
    # Setting the total length of the training text.
    def __len__(self):
        return len(self.textEncode)
    # Creating the training dataset.
    def __getitem__(self, index):
        # Getting the center word.
        centerWord = self.textEncode[index]
        # Getting the positive words according to the center word.                                                                                                                               
        positiveWords = self.textEncode[[i % len(self.textEncode) for i in (list(range(index - self.contextSize, index)) + list(range(index + 1, index + self.contextSize + 1)))]]
        # Sampling the negative words. 
        negativeWords = torch.multinomial(self.wordFreq, self.negativeSampling * positiveWords.shape[0], True)
        # Return the training data.                                                            
        return centerWord, positiveWords, negativeWords                                                                                                                     
    # Creating the necessary components of data generator.
    @staticmethod
    def generatorComponents(root, vocabularySize):
        # Opening the text file.
        textFile = open(root + '/train.txt')
        # Reading the text file and forming the text list.  
        text = textFile.read().split()
        # Getting the word which has the top 9999 appearing frequency and forming the vocabulary.                                                                                 
        vocab = dict(Counter(text).most_common(vocabularySize - 1))
        # Completing the vocabulary by replacing all the remaining words in text as '<unk>'.                                 
        vocab['<unk>'] = len(text) - np.sum(list(vocab.values()))
        # Getting the list for obtaining the word corresponding the index of the word.                                                   
        itos = [word for word in vocab.keys()]
        # Getting the dictionary for obtaining the index corresponding to the specific word in vocabulary, because the word would be inputted into the neural network as the index (one-hot encode).                                                                      
        stoi = {word:value for value, word in enumerate(itos)}
        # Getting the original sampling distribution of the words in the vocabulary.                                                      
        wordFreq = np.power(list(vocab.values()) / np.sum(list(vocab.values())), (3./4.))
        # Getting the final sampling distribution of the words in the vocabulary, this frequency is used to sampling the nagetive words from the vocabulary.
        wordFreq = torch.tensor(wordFreq / np.sum(wordFreq), dtype = torch.float32)
        # Returning the necessary components for generating the training data.                                
        return vocab, text, itos, stoi, wordFreq