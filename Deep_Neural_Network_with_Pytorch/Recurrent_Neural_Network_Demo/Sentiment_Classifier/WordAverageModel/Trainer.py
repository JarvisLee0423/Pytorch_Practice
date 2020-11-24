#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/24
#   File Name:      Trainer.py
#   Description:    This file is used to training the model.
#============================================================================================#

# Importing the necessary library.
import numpy as np
import logging
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from visdom import Visdom
from Utils.Config import argParse
from Utils.DataPreprocessor import dataLoader
from Model.WordAverageModel import WordAverageModelNN

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
def evaluator(model, loss, devSet):
    # Initializing the evaluation loss.
    evalLoss = []
    # Initializing the evaluation accuracy.
    evalAcc = []
    # Evaluating the model.
    for i, devData in enumerate(devSet):
        # Evaluating the model.
        prediction = model(devData.text)
        # Computing the loss.
        cost = loss(prediction, devData.label)
        # Storing the loss.
        evalLoss.append(cost.item())
        # Computing the accuracy.
        accuracy = (torch.round(torch.sigmoid(prediction)) == devData.label)
        accuracy = accuracy.sum().float() / len(accuracy)
        # Storing the accuracy.
        evalAcc.append(accuracy.item())
    # Returning the loss and accuracy.
    return np.sum(evalLoss) / len(evalLoss), np.sum(evalAcc) / len(evalAcc)

# Creating the training method.
def trainer(textField, trainSet, devSet):
    # Creating the logging.
    logging.basicConfig(filename = Cfg.logDir + f'/logging-{currentTime}.txt', filemode = 'a', level = logging.INFO, format = '%(asctime)s %(levelname)s %(message)s', datefmt = '%Y-%m-%d %H:%M:%S %p')
    # Logging the information.
    logging.info(f'''
        Vocabulary Size:        {Cfg.vs}
        Embedding Size:         {Cfg.es}
        Hidden Size:            {Cfg.hs}
        Class Size:             {Cfg.cs}
        Learning Rate:          {Cfg.lr}
        Adam Beta One:          {Cfg.beta1}
        Adam Beta Two:          {Cfg.beta2}
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
    vis = Visdom(env = 'WordAverageModel')
    # Creating the graph.
    lossGraph = vis.line(X = [0], Y = [0], opts = dict(legend = ['TrainingLoss', 'EvaluatingLoss'], xlabel = 'Epoches', ylabel = 'Loss', title = f'Training and Evaluating Loss - {currentTime}'), name = 'TrainingLoss')
    vis.line(X = [0], Y = [0], win = lossGraph, update = 'append', name = 'EvaluatingLoss')
    accGraph = vis.line(X = [0], Y = [0], opts = dict(legend = ['TrainingAcc', 'EvaluatingAcc'], xlabel = 'Epoches', ylabel = 'Acc', title = f'Training and Evaluating Acc - {currentTime}'), name = 'TrainingAcc')
    vis.line(X = [0], Y = [0], win = accGraph, update = 'append', name = 'EvaluatingAcc')
    # Creating the sequence to sequence model.
    model = WordAverageModelNN(Cfg.vs + 2, Cfg.es, Cfg.hs, Cfg.cs, textField.vocab.stoi[textField.pad_token]).to(device)
    # Customizing the initialized parameters of the embedding layer.
    # Getting the vocabulary as the vectors. 
    gloveVector = textField.vocab.vectors
    # Reinitializing the parameters of the embedding layer.
    model.embedding.weight.data.copy_(gloveVector)
    # Adding the '<unk>' and '<pad>' tokens into the parameters of the embedding layer.
    model.embedding.weight.data[textField.vocab.stoi[textField.pad_token]]
    model.embedding.weight.data[textField.vocab.stoi[textField.unk_token]]
    # Setting the optimizer.
    optimizer = optim.Adam(model.parameters(), lr = Cfg.lr, weight_decay = Cfg.wd, betas = [Cfg.beta1, Cfg.beta2])
    # Setting the loss function.
    loss = nn.BCEWithLogitsLoss()
    # Setting the list to storing the training loss.
    trainLosses = []
    # Setting the list to storing the training accuracy.
    trainAccs = []
    # Setting the list to storing the evaluating loss.
    evalLosses = []
    # Setting the list to storing the evaluating accuracy.
    evalAccs = []
    # Training the model.
    for epoch in range(Cfg.epoches):
        # Setting the list for storing the training loss and accuracy.
        trainLoss = []
        trainAcc = []
        # Setting the loading bar.
        with tqdm(total = len(trainSet), desc = f'Epoch {epoch + 1}/{Cfg.epoches}', unit = 'batches', dynamic_ncols = True) as pbars:
            for i, trainData in enumerate(trainSet):
                # Feeding the data into the model.
                prediction = model(trainData.text)
                # Computing the loss.
                cost = loss(prediction, trainData.label)
                # Storing the loss.
                trainLoss.append(cost.item())
                # Clearing the previous gradient.
                optimizer.zero_grad()
                # Applying the backward propagation.
                cost.backward()
                # Updating the parameters.
                optimizer.step()
                # Computing the accuracy.
                accuracy = (torch.round(torch.sigmoid(prediction)) == trainData.label)
                accuracy = accuracy.sum().float() / len(accuracy)
                # Storing the accurcy.
                trainAcc.append(accuracy.item())

                # Updating the loading bar.
                pbars.update(1)
                # Updating the training information.
                pbars.set_postfix_str(' - Train Loss %.4f - Train Acc %.4f' % (np.mean(trainLoss), np.mean(trainAcc)))
        # Closing the loading bar.
        pbars.close()
        # Printing the hint for evaluating.
        print('Evaluating...', end = ' ')   
        # Evalutaing the model.
        evalLoss, evalAcc = evaluator(model.eval(), loss, devSet)
        # Printing the evaluating information.
        print(' - Eval Loss %.4f - Eval Acc %.4f' % (evalLoss, evalAcc))
        # Storing the training and evaluating information.
        trainLosses.append(np.mean(trainLoss))
        trainAccs.append(np.mean(trainAcc))
        evalLosses.append(evalLoss)
        evalAccs.append(evalAcc)
        # Logging the information.
        logging.info('Epoch [%d/%d] -> Training: Loss [%.4f] - Acc [%.4f] || Evaluating: Loss [%.4f] - Acc [%.4f]' % (epoch + 1, Cfg.epoches, np.mean(trainLoss), np.mean(trainAcc), evalLoss, evalAcc))
        # Drawing the graph.
        vis.line(
            X = [k for k in range(1, len(trainLosses) + 1)],
            Y = trainLosses,
            win = lossGraph,
            update = 'new',
            name = 'TrainingLoss'
        )
        vis.line(
            X = [k for k in range(1, len(evalLosses) + 1)],
            Y = evalLosses,
            win = lossGraph,
            update = 'new',
            name = 'EvaluatingLoss'
        )
        vis.line(
            X = [k for k in range(1, len(trainAccs) + 1)],
            Y = trainAccs,
            win = accGraph,
            update = 'new',
            name = 'TrainingAcc'
        )
        vis.line(
            X = [k for k in range(1, len(evalAccs) + 1)],
            Y = evalAccs,
            win = accGraph,
            update = 'new',
            name = 'EvaluatingAcc'
        )
        # Giving the hint for saving the model.
        logging.info("Model Saved")
        # Saving the model.
        torch.save(model.train().state_dict(), Cfg.modelDir + f'/{currentTime}/WordAverageModel-Epoch{epoch + 1}.pt')
        # Converting the model state.
        model = model.train()
    # Saving the graph.
    vis.save(envs = ['WordAverageModel'])

# Setting the main function.
if __name__ == "__main__":
    # Generating the training data.
    textField, trainSet, devSet, testSet = dataLoader.IMDB(Cfg.dataDir, Cfg.vs, Cfg.bs, device)
    # Getting the command.
    cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit()' for quit): ")
    # Handling the command.
    while cmd != 'Exit()':
        # Handling the command.
        if cmd == 'T':
            # Training the model.
            trainer(textField, trainSet, devSet)
            cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit()' for quit): ")
        elif cmd == 'E':
            try:
                # Creating the sequence to sequence model.
                model = WordAverageModelNN(Cfg.vs + 2, Cfg.es, Cfg.hs, Cfg.cs, textField.vocab.stoi[textField.pad_token]).to(device)
                # Loading the model.
                model.load_state_dict(torch.load(Cfg.modelDir + f'/{currentTime}/WordAverageModel-Epoch{Cfg.epoches}.pt'))
                # Sending the model into the corresponding computer device.
                model = model.to(device)
                # Testing the model.
                testLoss, testAcc = evaluator(model.eval(), nn.BCEWithLogitsLoss(), testSet)
                # Printing the testing result.
                print("The testing: Loss = " + str(testLoss) + " || Acc = " + str(testAcc))
                # Getting the input sentence.
                sentence = input("Please input one sentiment sentence ('Exit() for quit'): ")
                while sentence != 'Exit()':
                    # Getting the words from the sentence.
                    words = [word for word in sentence.split()]
                    # Getting the index of the word.
                    wordsIndex = [textField.vocab.stoi[word] for word in words]
                    # Sending the words' index into the corresponding device.
                    wordsIndex = torch.LongTensor(wordsIndex).to(device).unsqueeze(1)
                    # Getting the prediction.
                    prediction = int(torch.sigmoid(model(wordsIndex)).item())
                    # Giving the predicted result.
                    if prediction == 0:
                        print("The sentence is negative sentiment! :(")
                    else:
                        print("The sentence is positive sentiment! :)")
                    # Getting the input sentence.
                    sentence = input("Please input one sentiment sentence ('Exit() for quit'): ")
            except:
                # Giving the hint.
                print("There are not any trained model, please train one first!!!")
                sentence = 'T'
            cmd = sentence
        else:
            cmd = input("Invalid Input! Please input the command ('T' for training, 'E' for evaluating, 'Exit() for quit'): ")