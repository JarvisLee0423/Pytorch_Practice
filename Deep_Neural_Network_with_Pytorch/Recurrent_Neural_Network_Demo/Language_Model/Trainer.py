#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/24
#   File Name:      Trainer.py
#   Description:    This file is used to training the model.
#============================================================================================#

# Importing the necessary library.
import os
import time
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from visdom import Visdom
from Utils.Config import argParse
from Utils.DataPreprocessor import dataLoader
from Model.LanguageModel import LanguageModelNN

# Creating the configurator.
Cfg = argParse()

# Creating the model directory.
if not os.path.exists(Cfg.modelDir):
    os.mkdir(Cfg.modelDir)
# Creating the logging directory.
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
def evaluator(model, loss, devSet, batchSize, hiddenSize):
    # Initializing the evaluation loss.
    evalLoss = []
    # Initializing the evaluation accuracy.
    evalAcc = []
    # Initializing the hidden.
    hidden = model.initHidden(batchSize, hiddenSize, requireGrad = False)
    # Evaluating the model.
    for i, evalData in enumerate(devSet):
        # Remembering the historical hidden.
        hidden = model.splitHiddenHistory(hidden)
        # Evaluating the model.
        prediction, hidden = model(evalData.text, hidden)
        # Computing the loss.
        cost = loss(prediction, evalData.target.view(-1))
        # Storing the loss.
        evalLoss.append(cost.item())
        # Computing the accuracy.
        accuracy = (torch.argmax(prediction, 1) == evalData.target.view(-1))
        accuracy = accuracy.sum().float() / len(accuracy)
        # Storing the accuracy.
        evalAcc.append(accuracy.item())
    # Returning the loss and accuracy.
    return np.sum(evalLoss) / len(evalLoss), np.sum(evalAcc) / len(evalAcc)

# Defining the training method.
def trainer(trainSet, devSet):
    # Getting the current time.
    currentTime = time.strftime('%Y-%m-%d-%H-%M-%S')
    # Setting the logging.
    logging.basicConfig(filename = Cfg.logDir + f'/logging-{currentTime}.txt', filemode = 'a', level = logging.INFO, format = '%(asctime)s %(levelname)s %(message)s', datefmt = '%Y-%m-%d %H:%M:%S %p')
    # Logging the information.
    logging.info(f'''
        Vocabulary Size:            {Cfg.vs}
        Embedding Size:             {Cfg.es}
        Hidden Size:                {Cfg.hs}
        Batch Size:                 {Cfg.bs}
        Learning Rate:              {Cfg.lr}
        Epoches:                    {Cfg.epoches}
        Random Seed:                {Cfg.seed}
        GPU ID:                     {Cfg.GPUID}
        Model Directory:            {Cfg.modelDir}
        Log Directory:              {Cfg.logDir}
        Data Directory:             {Cfg.dataDir}
    ''')
    # Setting the visdom.
    vis = Visdom(env = 'LanguageModel')
    # Setting the graph.
    lossGraph = vis.line(X = [0], Y = [0], opts = dict(legend = ['TrainingLoss', 'EvaluatingLoss'], xlabel = 'Epoches', ylabel = 'Loss', title = f'Training and Evaluating Loss for {currentTime}'), name = 'TrainingLoss')
    vis.line(X = [0], Y = [0], win = lossGraph, update = 'append', name = 'EvaluatingLoss')
    accGraph = vis.line(X = [0], Y = [0], opts = dict(legend = ['TrainingAcc', 'EvaluatingAcc'], xlabel = 'Epoches', ylabel = 'Acc', title = f'Training and Evaluating Acc for {currentTime}'), name = 'TrainingAcc')
    vis.line(X = [0], Y = [0], win = accGraph, update = 'append', name = 'EvaluatingAcc')
    # Setting the list for training loss.
    trainLosses = []
    # Setting the list for training accuracy.
    trainAccs = []
    # Setting the list for evaluating loss.
    evalLosses = []
    # Setting the list for evaluating accuracy.
    evalAccs = []
    # Creating the model.
    model = LanguageModelNN(Cfg.vs, Cfg.es, Cfg.hs)
    # Sending the model into the specific device.
    model = model.to(device)
    # Creating the loss function.
    loss = nn.CrossEntropyLoss()
    # Creating the optimizer.
    optimizer = optim.Adam(model.parameters(), lr = Cfg.lr)
    # Training the model.
    for epoch in range(Cfg.epoches):
        # Initializing the training loss.
        trainLoss = []
        # Initializing the training accuracy.
        trainAcc = []
        # Initializing the hidden.
        hidden = model.initHidden(Cfg.bs, Cfg.hs)
        # Setting the loading bar.
        with tqdm(total = len(trainSet), desc = f'Epoch {epoch + 1}/{Cfg.epoches}', unit = 'batches', dynamic_ncols = True, ) as pbars:
            # Training the model.
            for i, trainData in enumerate(trainSet):
                # Remembering the historical hidden.
                hidden = model.splitHiddenHistory(hidden)
                # Feeding the data into the model.
                prediction, hidden = model(trainData.text, hidden)
                # Getting the value of the loss.
                cost = loss(prediction, trainData.target.view(-1))
                # Storing the cost.
                trainLoss.append(cost.item())
                # Clearing the previous gradient.
                optimizer.zero_grad()
                # Applying the backward propagation.
                cost.backward()
                # Updating the parameters.
                optimizer.step()
                # Computing the accuracy.
                accuracy = (torch.argmax(prediction, 1) == trainData.target.view(-1))
                accuracy = accuracy.sum().float() / len(accuracy)
                # Storing the accuracy.
                trainAcc.append(accuracy.item())
                
                # Updating the loading bar.
                pbars.update(1)
                # Updating the training information.
                pbars.set_postfix_str('Train Loss: %.4f - Train Acc: %.4f' % (np.mean(trainLoss), np.mean(trainAcc)))
        # Closing the loading bar.
        pbars.close()
        # Printing the evaluating hint.
        print('Evaluating...', end = ' ')
        # Evaluating the model.
        evalLoss, evalAcc = evaluator(model.eval(), loss, devSet, Cfg.bs, Cfg.hs)
        # Printing the evluating information.
        print('- Eval Loss: %.4f - Eval Acc: %.4f' % (evalLoss, evalAcc))
        # Storing the training and evaluating information.
        trainLosses.append(np.mean(trainLoss))
        trainAccs.append(np.mean(trainAcc))
        evalLosses.append(evalLoss)
        evalAccs.append(evalAcc)
        # Logging the training and evaluating information.
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
        # Saving the model.
        torch.save(model.state_dict(), Cfg.modelDir + f'/LanguageModel-Epoch{epoch + 1}.pt')
        logging.info("Model Saved")
        # Converting the model mode.
        model.train()
    # Saving the visdom.
    vis.save(envs = ['LanguageModel'])

# Training the model.
if __name__ == "__main__":
    # Generating the data.
    vocab, trainSet, devSet, testSet = dataLoader.Datasets(Cfg.dataDir, device, Cfg.vs, Cfg.bs)
    cmd = input("Please input the command ('T' for training, 'E' for evaluation, 'Exit()' for quit): ")
    # Handling the command.
    while cmd != 'Exit()':
        if cmd == 'T':
            # Training the model.
            trainer(trainSet, devSet)
            cmd = input("Please input the command ('T' for training, 'E' for evaluation, 'Exit()' for quit): ")
        elif cmd == 'E':
            try:
                # Creating the model.
                model = LanguageModelNN(Cfg.vs, Cfg.es, Cfg.hs)
                # Loading the model.
                model.load_state_dict(torch.load(Cfg.modelDir + f'/LanguageModel-Epoch{Cfg.epoches}.pt'))
                # Sending the model into the corresponding device.
                model = model.to(device)
                # Testing the model by perplexity.
                testLoss, _ = evaluator(model.eval(), nn.CrossEntropyLoss(), testSet, Cfg.bs, Cfg.hs)
                # Printing the perplexity.
                print("The perplexity is: " + str(np.exp(testLoss)))
                # Getting the first word of the sentence.
                word = input("Please input the first word ('Exit()' for quit): ")
                while word != 'Exit()':
                    # Getting the first word.
                    wordIndex = vocab.stoi.get(str.lower(word), vocab.stoi.get('<unk>'))
                    # Initializing the sentence.
                    sentence = [vocab.itos[wordIndex]]
                    # Generating the input data.
                    data = torch.tensor([[wordIndex]]).to(device)
                    # Initializing the hidden.
                    hidden = model.splitHiddenHistory(model.initHidden(1, Cfg.hs, False))
                    # Getting the prediction.
                    for i in range(100):
                        # Getting the prediction.
                        prediction, hidden = model(data, hidden)
                        # Getting the index of the predicted word.
                        index = torch.multinomial(prediction.squeeze().exp().cpu(), 1)[0]
                        # Storing the predicted word.
                        sentence.append(vocab.itos[index])
                        # Getting another data.
                        data.fill_(index)
                    # Printing the predicted sentence.
                    print("The predicted sentence is: " + " ".join(sentence))
                    # Getting another first word of the sentence.
                    word = input("Please input the first word ('Exit()' for quit): ")
            except:
                # Giving the hint.
                print("There are not any trained model, please train one first!!!")
                word = 'T'
            cmd = word
        else:
            cmd = input("Invalid Input! Please input the command ('T' for training, 'E' for evaluation, 'Exit()' for quit): ") 