#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/25
#   File Name:      Trainer.py
#   Description:    This file is used to training the model.
#============================================================================================#

# Importing the necessary library.
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tud
import scipy.spatial
from tqdm import tqdm
from visdom import Visdom
from Utils.Config import argParse
from Utils.DataPreprocessor import dataLoader
from Model.SkipGramModel import SkipGramNN

# Getting the configurator.
Cfg = argParse()

# Setting the current time.
if Cfg.currentTime != -1:
    currentTime = Cfg.currentTime
else:
    currentTime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

# Creating the model directory.
if not os.path.exists(Cfg.modelDir):
    os.mkdir(Cfg.modelDir)
if not os.path.exists(Cfg.modelDir + f'/{currentTime}'):
    os.mkdir(Cfg.modelDir + f'/{currentTime}')
# Creating the log directory.
if not os.path.exists(Cfg.logDir):
    os.mkdir(Cfg.logDir)

# Setting the device and random seed.
if torch.cuda.is_available():
    # Setting the device.
    device = 'cuda'
    # Fixing the device.
    if Cfg.GPUID != -1:
        torch.cuda.set_device(Cfg.GPUID)
    # Fixing the random seed.
    np.random.seed(Cfg.seed)
    torch.cuda.manual_seed(Cfg.seed)
else:
    # Setting the device.
    device = 'cpu'
    # Fixing the random seed.
    np.random.seed(Cfg.seed)
    torch.manual_seed(Cfg.seed)

# Defining the evaluation method.
def evaluator(embeddingMatrix, testingWord, itos, stoi):
    # Obtaning the index of the testing word from the vocabulary.
    wordIndex = stoi.get(testingWord, stoi.get('<unk>'))
    # Getting the word embedding of the testing word.                                                    
    wordEmbedding = embeddingMatrix[wordIndex]
    # Computing the consine similarity of the testing word and other words in embedding level.                                                              
    cosineDistance = np.array([scipy.spatial.distance.cosine(e, wordEmbedding) for e in embeddingMatrix])
    # Returning the top ten most similar words of the testing word.   
    return [itos[index] for index in cosineDistance.argsort()[:30]]

# Defining the training method.
def trainer(trainSet):
    # Creating the logging.
    logging.basicConfig(filename = Cfg.logDir + f'/logging-{currentTime}.txt', filemode = 'a', level = logging.INFO, format = '%(asctime)s %(levelname)s %(message)s', datefmt = '%Y-%m-%d %H:%M:%S %p')
    # Logging the information.
    logging.info(f'''
        Context Size:           {Cfg.cs}
        Negative Sampling:      {Cfg.ns}
        Vocabulary Size:        {Cfg.vs}
        Embedding Size:         {Cfg.es}
        Learning Rate:          {Cfg.lr}
        Weight Decay:           {Cfg.wd}
        Batch Size:             {Cfg.bs}
        Epoches:                {Cfg.epoches}
        Random Seed:            {Cfg.seed}
        GPU ID:                 {Cfg.GPUID}
        Model Directory:        {Cfg.modelDir}
        Log Directory:          {Cfg.logDir}
        Dataset Directory:      {Cfg.dataDir}
    ''')
    # Creating the visdom.
    vis = Visdom(env = 'SkipGramModel')
    # Creating the graph.
    lossGraph = vis.line(X = [0], Y = [0], opts = dict(legend = ['TrainingLoss'], xlabel = 'Epoches', ylabel = 'Loss', title = f'Training and Evaluating Loss - {currentTime}'), name = 'TrainingLoss')
    # Creating the model.
    model = SkipGramNN(Cfg.vs, Cfg.es)
    # Sending the model to the correct device.
    model.to(device)
    # Setting the optimizer.                                                  
    optimizer = optim.Adam(model.parameters(), lr = Cfg.lr, weight_decay = Cfg.wd) 
    # Initializing the previous cost.                           
    previousCost = 0  
    # Setting the training loss.
    trainLosses = []
    # Training the model.
    for epoch in range(Cfg.epoches):
        # Initializing the cost.
        trainLoss = []
        # Setting the loading bar.
        with tqdm(total = len(trainSet), desc = f'Epoch {epoch + 1}/{Cfg.epoches}', unit = 'batches', dynamic_ncols = True) as pbars:
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
                trainLoss.append(loss.item())
                # Clearing the previous gradient.                                                      
                optimizer.zero_grad()
                # Appling the backword propagation.                                                           
                loss.backward()
                # Updating the parameters.                                                                
                optimizer.step()

                # Updating the loading bar.
                pbars.update(1)
                # Updating the training information.
                pbars.set_postfix_str(' - Train Loss %.4f' % (np.mean(trainLoss)))
        # Closing the loading bar.                                                             
        pbars.close()
        # Storing the training loss.
        trainLosses.append(np.mean(trainLoss))
        # Logging the information.
        logging.info('Epoch [%d/%d] -> Training: Loss [%.4f]' % (epoch + 1, Cfg.epoches, np.mean(trainLoss)))
        # Drawing the graph.
        vis.line(
            X = [k for k in range(1, len(trainLosses) + 1)],
            Y = trainLosses,
            win = lossGraph,
            update = 'new',
            name = 'TrainingLoss'
        )
        # Storing the model.                                                                                        
        torch.save(model.state_dict(), Cfg.modelDir + f'/{currentTime}/SkipGram-Epoch{epoch + 1}.pt')
        # Providing the hint for saving model.   
        logging.info("Model Saved")

# Training the model.
if __name__ == "__main__":
    # Getting the necessary components of data generator.
    vocab, text, itos, stoi, wordFreq = dataLoader.generatorComponents(Cfg.dataDir, Cfg.vs)
    # Generating the training set.                                         
    trainSet = tud.DataLoader(dataLoader(text, itos, stoi, wordFreq, Cfg.cs, Cfg.ns), batch_size = Cfg.bs, shuffle = True)                                                                                              
    cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit()' for quit): ")
    while cmd != 'Exit()':
        if cmd == 'T':
            # Training the model.
            trainer(trainSet)
            cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit()' for quit): ")
        elif cmd == 'E':                                                                                                                 
            try:
                # Loading the paramters.
                params = torch.load(Cfg.modelDir + f'/{currentTime}/SkipGram-Epoch{Cfg.epoches}.pt')
                # Getting the testing words.
                word = input("Please input a word ('Exit()' for quit): ")
                # Indicating whether applying the testing or not.                                                                               
                while word != 'Exit()':
                    # Getting the parameters.              
                    embeddingMatrix = params['inputEmbedding.weight'].cpu().numpy()
                    # Converting the testing word into lowercase.                                                                              
                    word = str.lower(word)
                    # Printing the testing result.                                                                                                   
                    print("The similar words of " + word + " are : " + " ".join(evaluator(embeddingMatrix, word, itos, stoi)))
                    # Getting another testing word.            
                    word = input("Please input a word ('Exit()' for quit): ")                                                                       
            except:
                # Giving the hint.
                print("Please training a model first!!!")
                # Applying the training.                                                                                       
                word = 'T'                                                                                                                      
            cmd = word
        else:
            cmd = input("Invalid Input! Please input the command ('T' for training, 'E' for evaluating, 'Exit()' for quit): ")