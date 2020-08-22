#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/08/21
#   Project Name:       SkipGram.py
#   Description:        Build the word embedding model by applying the skip-gram algorithm
#                       with pytorch.
#   Model Description:  Input               ->  Center word
#                                           ->  2 * Context size positive words 
#                                               for each center word
#                                           ->  Randomly selected negative words 
#                                               for each pair of center word and positive word
#                       Embedding Layer     ->  Convert vocabulary size input into embedding
#                                               size
#                       Output              ->  Sigmoid value to indicate whether the input 
#                                               pair of words is positive pair(1) or negative
#                                               pair(0).
#                       Train Target        ->  Get the weight of the embedding layer. 
#============================================================================================#

# Importing the necessary libraries.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as tud
import numpy as np
import scipy.spatial
from collections import Counter

# Fixing the computation device and random seed.
# Checking whether the computer supporting the 
if torch.cuda.is_available:
    # Fixing the GPU.    
    torch.cuda.set_device(0)
    # Fixing the random seed.   
    torch.cuda.manual_seed(1)
    # Applying the GPU.
    device = 'cuda'
else:
    # Fixing the random seed.
    torch.manual_seed(1)
    # Applying the CPU.
    device = 'cpu'

# Setting the hyperparameters.
# The context size of each center word.
contextSize = 3
# The number of negative words for each pair of positive word and center word.
negativeSampling = 10
# The size of the vocabulary.
vocabularySize = 10000
# The size of the word embedding encode for each word.
embeddingSize = 100
# The value of the learning rate.
learningRate = 0.01
# The value of the bacth size.
batchSize = 128
# The value of the training epoch.
epoch = 2

# Creating the training data generator.
class dataGenerator(tud.Dataset):
    # Creating the constructor.
    def __init__(self, text, itos, stoi, wordFreq):
        # Inheritting the super class.
        super(dataGenerator, self).__init__()
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
        positiveWords = self.textEncode[[i % len(self.textEncode) for i in (list(range(index - contextSize, index)) + list(range(index + 1, index + contextSize + 1)))]]
        # Sampling the negative words. 
        negativeWords = torch.multinomial(self.wordFreq, negativeSampling * positiveWords.shape[0], True)
        # Return the training data.                                                            
        return centerWord, positiveWords, negativeWords                                                                                                                     
    # Creating the necessary components of data generator.
    @staticmethod
    def generatorComponents():
        # Opening the text file.
        textFile = open('./Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Word_Embedding/train.txt')
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
    # Defining the training method.
    @staticmethod
    def trainer(model, optimizer, trainSet, epoch = epoch, vocabularySize = vocabularySize, embeddingSize = embeddingSize):
        # Indicate whether the model is correct.
        assert type(model) != type(SkipGramNN)
        # Sending the model to the correct device.
        model.to(device)
        # Initializing the previous cost.                           
        previousCost = 0  
        # Training the model.
        for epoch in range(epoch):
            # Initializing the cost.
            cost = []
            # Getting the training data.                                                                          
            for i, (centerWords, positiveWords, negativeWords) in enumerate(trainSet):
                # Sending the center words into the corresponding device.          
                centerWords = centerWords.to(device)
                # Sending the positive words into the corresponding device.                                          
                positiveWords = positiveWords.to(device)
                # Sending the negative words into the corresponding device.                                        
                negativeWords = negativeWords.to(device)
                # Getting the loss.                                       
                loss = model(centerWords, positiveWords, negativeWords)
                # Storing the loss.                         
                cost.append(loss.item())
                # Clearing the previous gradient.                                                      
                optimizer.zero_grad()
                # Appling the backword propagation.                                                           
                loss.backward()
                # Updating the parameters.                                                                
                optimizer.step()                                                                
                if i % 100 == 0:
                    print("The loss of training iteration " + str(i) + " is: " + str(loss.item()))
            # Getting the training cost.
            cost = np.sum(cost) / len(cost) 
            # Printing the training cost.                                                                                  
            print("The training loss of epoch " + str(epoch + 1) + " is: " + str(cost))
            # Indicating whether storing the model or not.                                                         
            if previousCost == 0 or cost < previousCost:
                # Storing the model.                                                                                        
                torch.save(model.state_dict(), './Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Word_Embedding/SkipGram.pt')
                # Providing the hint for saving model.   
                print("Model Saved")
            # Updating previous cost.                                                                                                            
            previousCost = cost                                                                                                                                                              
    # Defining the evaluation method.
    @staticmethod
    def evaluator(embeddingMatrix, testingWord, itos, stoi):
        # Obtaning the index of the testing word from the vocabulary.
        wordIndex = stoi.get(testingWord, stoi.get('<unk>'))
        # Getting the word embedding of the testing word.                                                    
        wordEmbedding = embeddingMatrix[wordIndex]
        # Computing the consine similarity of the testing word and other words in embedding level.                                                              
        cosineDistance = np.array([scipy.spatial.distance.cosine(e, wordEmbedding) for e in embeddingMatrix])
        # Returning the top ten most similar words of the testing word.   
        return [itos[index] for index in cosineDistance.argsort()[:10]]                                         

# Training the model.
if __name__ == "__main__":
    pass
    # # Getting the necessary components of data generator.
    # vocab, text, itos, stoi, wordFreq = dataGenerator.generatorComponents()
    # # Generating the training set.                                         
    # trainSet = tud.DataLoader(dataGenerator(text, itos, stoi, wordFreq), batch_size = batchSize, shuffle = True)
    # # Creating the model.
    # model = SkipGramNN(vocabularySize, embeddingSize)
    # # Setting the optimizer.                                                  
    # optimizer = optim.Adam(model.parameters(), lr = learningRate, weight_decay = 0.00005)                                                                                              
    # cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit()' for quit): ")
    # while cmd != 'Exit()':
    #     if cmd == 'T':
    #         # Training the model.
    #         SkipGramNN.trainer(model, optimizer, trainSet)
    #         cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit()' for quit): ")
    #     elif cmd == 'E':
    #         # Getting the testing words.
    #         word = input("Please input a word ('Exit()' for quit): ")
    #         # Indicating whether applying the testing or not.                                                                               
    #         while word != 'Exit()':                                                                                                                 
    #             try:
    #                 # Loading the paramters.
    #                 params = torch.load('./Deep_Neural_Network_with_Pytorch/Recurrent_Neural_Network_Demo/Word_Embedding/SkipGram.pt')
    #                 # Getting the parameters.              
    #                 embeddingMatrix = params['inputEmbedding.weight'].cpu().numpy()
    #                 # Converting the testing word into lowercase.                                                                              
    #                 word = str.lower(word)
    #                 # Printing the testing result.                                                                                                   
    #                 print("The similar words of " + word + " are : " + " ".join(SkipGramNN.evaluator(embeddingMatrix, word, itos, stoi)))
    #                 # Getting another testing word.            
    #                 word = input("Please input a word ('Exit()' for quit): ")                                                                       
    #             except:
    #                 # Giving the hint.
    #                 print("Please training a model first!!!")
    #                 # Applying the training.                                                                                       
    #                 word = 'T'                                                                                                                      
    #                 break
    #         cmd = word
    #     else:
    #         cmd = input("Invalid Input! Please input the command ('T' for training, 'E' for evaluating, 'Exit()' for quit): ")