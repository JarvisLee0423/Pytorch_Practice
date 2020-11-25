#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/11/25
#   Project Name:       SkipGram.py
#   Description:        Build the word embedding model by applying the skip-gram algorithm
#                       with pytorch.
#   Model Description:  Input               ->  Center word
#                                           ->  2 * Context size positive words 
#                                               for each center word
#                                           ->  Randomly selected negative words 
#                                               for each pair of center word and positive word
#                       Embedding Layer     ->  Converting vocabulary size input into 
#                                               embedding size
#                       Output              ->  Sigmoid value to indicate whether the input 
#                                               pair of words is positive pair(1) or negative
#                                               pair(0)
#                       Train Target        ->  Getting the weight of the embedding layer
#============================================================================================#

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Creating the model.
class SkipGramNN(nn.Module):
    # Creating the constructor.
    def __init__(self, vocabularySize, embeddingSize):
        # Inheritting the super class.
        super(SkipGramNN, self).__init__()
        # Setting the embedding layer.                                              
        self.inputEmbedding = nn.Embedding(vocabularySize, embeddingSize)
        # Setting the embedding layer.              
        self.outputEmbedding = nn.Embedding(vocabularySize, embeddingSize)                  
    # Defining the forward propagation.
    def forward(self, centerWords, positiveWords, negativeWords):
        # Embedding the center words. [batchSize] -> [batchSize, embeddingSize]
        centerWords = self.inputEmbedding(centerWords)
        # Embedding the positive words. [bathSize, 2 * contextSize] -> [batchSize, 2 * contextSize, embeddingSize]                                    
        positiveWords = self.outputEmbedding(positiveWords)
        # Embedding the negative words. [batchSize, 2 * contextSize * negativeSampling] -> [batchSize, 2 * contextSize * negativeSampling, embeddingSize]
        negativeWords = self.outputEmbedding(negativeWords)
        # Adding one more dimension to the center words. For Skip-gram algorithm, the loss is the sigmoid value of the (theta_p.T * e_c) or (theta_n.T * e_c). Therefore, it is necessary to make the dimension of the center words match the dimension of positive words or negative words to complete the multiplication. [batchSize, embeddingSize] -> [batchSize, embeddingSize, 1]
        centerWords = centerWords.unsqueeze(2)
        # Getting the prediction of the positive pair. [batchSize, 2 * contextSize, embeddingSize] * [batchSize, embeddingSize, 1] -> [batchSize, 2 * contextSize, 1] -> [batchSize, 2 * contextSize] -> [batchSize]
        posPredict = F.logsigmoid(torch.bmm(positiveWords, centerWords).squeeze()).sum(1)
        # Getting the prediction of the negative pair. [batchSize, 2 * contextSize * negativeSampling, embeddingSize] * [batchSize, embeddingSize, 1] -> [batchSize, 2 * contextSize * negativeSampling, 1] -> [batchSize, 2 * contextSize * negativeSampling] -> [batchSize]
        negPredict = F.logsigmoid(torch.bmm(negativeWords, -centerWords).squeeze()).sum(1)
        # Computing the loss.
        loss = -(posPredict + negPredict)
        # Returing the loss.                                         
        return loss.mean()