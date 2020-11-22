#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/22
#   File Name:      Trainer.py
#   Description:    This file is used to training the model.
#============================================================================================#

# Importing the necessary library.
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from visdom import Visdom
from PIL import Image
from Utils.Config import argParse
from Utils.DataPreprocessor import dataLoader
from Model.ResnetTransferModel import ResnetTransferModelNN

# Creating the configurator.
Cfg = argParse()

# Creating the model directory.
if not os.path.exists(Cfg.modelDir):
    os.mkdir(Cfg.modelDir)

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

# Defining the method to evaluating the model.
def evaluator(model, loss, devSet):
    # Initializing the evaluating loss.
    evalLoss = []
    # Initializing the evaluating accuracy.
    evalAcc = []
    # Evaluating the model.
    for i, (data, label) in enumerate(devSet):
        # Preparing the data.
        data = data.to(device)
        label = label.to(device)
        # Computing the prediction.
        prediction = model(data)
        # Computing the loss.
        cost = loss(prediction, label)
        # Storing the loss.
        evalLoss.append(cost.item())
        # Computing the accuracy.
        accuracy = (torch.argmax(prediction, 1) == label)
        accuracy = accuracy.sum().float() / len(accuracy)
        # Storing the accuracy.
        evalAcc.append(accuracy.item())
        # Returning the evaluating result.
        return np.sum(evalLoss) / len(evalLoss), np.sum(evalAcc) / len(evalAcc)

# Defining the method to training the data.
def trainer(trainSet, devSet):
    # Setting the visdom.
    vis = Visdom(env = 'ResnetTransferModel')
    # Setting the graphs.
    lossGraph = vis.line(X = [0], Y = [0], opts = dict(legend = ['TrainingLoss', 'EvaluatingLoss'], xlabel = 'Epoches', ylabel = 'Loss', title = 'Training and Evaluating Loss'), name = 'TrainingLoss')
    vis.line(X = [0], Y = [0], win = lossGraph, update = 'append', name = 'EvaluatingLoss')
    accGraph = vis.line(X = [0], Y = [0], opts = dict(legend = ['TrainingAcc', 'EvaluatingAcc'], xlabel = 'Epoches', ylabel = 'Acc', title = 'Training and Evaluating Acc'), name = 'TrainingAcc')
    vis.line(X = [0], Y = [0], win = accGraph, update = 'append', name = 'EvaluatingAcc')
    # Setting the logging.
    logging.basicConfig(filename = Cfg.logDir + '/logging.txt', filemode = 'a', level = logging.INFO, format = '%(asctime)s %(levelname)s %(message)s', datefmt = '%Y-%D %H:%M:%S %p')
    # Logging the information.
    logging.info(f'''
        Class Size:         {Cfg.cs}
        Learning Rate:      {Cfg.lr}
        Batch Size:         {Cfg.bs}
        Epoches:            {Cfg.epoches}
        Random Seed:        {Cfg.seed}
        GPU ID:             {Cfg.GPUID}
        Model Directory:    {Cfg.modelDir}
        Log Directory:      {Cfg.logDir}
        Dataset Directory:  {Cfg.dataDir}
    ''')
    # Creating the model.
    model = ResnetTransferModelNN.ResnetInitialization(Cfg.cs)
    # Sending the model into the corresponding device.
    model = model.to(device)
    # Creating the optimizer.
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = Cfg.lr, betas = [0.9, 0.999])
    # Creating the learning rate scheduler.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = len(trainSet), T_mult = 1, eta_min = 0)
    # Creating the loss function.
    loss = nn.CrossEntropyLoss()
    # Initializing the evaluating accuracy.
    evalAccs = []
    # Initializing the evaluating loss.
    evalLosses = []
    # Initializing the training accuracy.
    trainAccs = []
    # Initializing the training accuracy.
    trainLosses = []
    # Training the model.
    for epoch in range(Cfg.epoches):
        # Initializing the train accuracy.
        trainAcc = []
        # Initializing the train loss.
        trainLoss = []
        # Setting the loading bar.
        with tqdm(total = len(trainSet), desc = f'Epoch {epoch + 1}/{Cfg.epoches}', unit = 'batches', ncols = 100) as pbars:
            # Getting the training data.
            for i, (data, label) in enumerate(trainSet):
                # Preparing the data.
                data = data.to(device)
                label = label.to(device)
                # Computing the prediction.
                prediction = model(data)
                # Computing the loss.
                cost = loss(prediction, label)
                # Storing the loss.
                trainLoss.append(cost.item())
                # Clearing the previous gradient.
                optimizer.zero_grad()
                # Computing the backward.
                cost.backward()
                # Updating the parameters.
                optimizer.step()
                # Computing the accuracy.
                accuracy = (torch.argmax(prediction, 1) == label)
                accuracy = accuracy.sum().float() / len(accuracy)
                # Storing the accuracy.
                trainAcc.append(accuracy.item())

                # Updating the loading bar.
                pbars.update(1)
        # Closing the loading bar.
        pbars.close()
        # Printing the hint.
        print('Evaluating...')
        # Evaluating the model.
        evalLoss, evalAcc = evaluator(model.eval(), loss, devSet)
        # Storing the evaluating.
        evalAccs.append(evalAcc)
        evalLosses.append(evalLoss)
        # Storing the training.
        trainAccs.append(np.sum(trainAcc) / len(trainAcc))
        trainLosses.append(np.sum(trainLoss) / len(trainLoss))
        # Logging the information.
        logging.info('Epoch [%d/%d] -> Training: Loss [%.4f] - Acc [%.4f] || Evaluating: Loss [%.4f] - Acc [%.4f]' % (epoch + 1, Cfg.epoches, np.sum(trainLoss) / len(trainLoss), np.sum(trainAcc) / len(trainAcc), evalLoss, evalAcc))
        # Drawing the graphs.
        vis.line(
            Y = evalAccs,
            X = [k for k in range(1, len(evalAccs) + 1)],
            win = accGraph,
            update = 'new',
            name = 'EvaluatingAcc'
        )
        vis.line(
            Y = trainAccs,
            X = [k for k in range(1, len(trainAccs) + 1)],
            win = accGraph,
            update = 'new',
            name = 'TrainingAcc'
        )
        vis.line(
            Y = evalLosses,
            X = [k for k in range(1, len(evalLosses) + 1)],
            win = lossGraph,
            update = 'new',
            name = 'EvaluatingLoss'
        )
        vis.line(
            Y = trainLosses,
            X = [k for k in range(1, len(trainLosses) + 1)],
            win = lossGraph,
            update = 'new',
            name = 'TrainingLoss'
        )
        torch.save(model.train().state_dict(), Cfg.modelDir + f'/ResnetTransferModel-Epoch{epoch + 1}.pt')
        logging.info("Model Saved")
        # Selecting the best model.
        if evalAcc < max(evalAccs):
            scheduler.step()
            logging.info("Learning Rate Decay -> " + str(scheduler.get_last_lr()))
        # Converting the model mode.
        model.train()

# Training the model.
if __name__ == "__main__":
    # Getting the training data.
    trainSet, devSet = dataLoader.Datasets(Cfg.dataDir, Cfg.bs)
    # Getting the data dictionary.
    dataDict = {'trainSet': trainSet, 'devSet': devSet}
    for each in ['trainSet', 'devSet']:
        # Reading the data.
        for _, (data, _) in enumerate(dataDict[each]):
            # Getting the transforming method.
            transform = transforms.ToPILImage()
            # Getting each image.
            for j in range(len(data)):
                # Getting the mean and std.
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                # Recovering the image.
                data[j][0] = data[j][0] * std[0] + mean[0]
                data[j][1] = data[j][1] * std[1] + mean[1]
                data[j][2] = data[j][2] * std[2] + mean[2]
                # Transforming the tensor into image.
                image = transform(data[j])
                # Showing the image.
                plt.imshow(image)
                plt.show()
                # Getting the cmd.
                cmd = input("'Exit' for quitting showing the image: ")
                # Handling the command.
                if cmd == 'Exit':
                    break
            if cmd == 'Exit':
                break
    # Getting the command.
    cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
    # Handling the command.
    while cmd != 'Exit':
        if cmd == 'T':
            # Training the model.
            trainer(trainSet, devSet)
            cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
        elif cmd == 'E':
            try:
                # Getting the output label.
                labels = ['Class_1', 'Class_2', 'Class_3']
                # Loading the model.
                model = ResnetTransferModelNN.ResnetInitialization(Cfg.cs)
                model.load_state_dict(torch.load(Cfg.modelDir + f'/ResnetTransferModel-Epoch{Cfg.epoches}.pt'))
                # Sending the model into corresponding device.
                model = model.to(device)
                # Converting the model mode.
                model.eval()
                # Getting the data.
                for _, (data, label) in enumerate(devSet):
                    # Getting the data.
                    for j in range(len(data)):
                        # Preparing the testing data.
                        testData = data[j].to(device)
                        # Getting the prediction.
                        prediction = model(testData.unsqueeze(0))
                        # Getting the predicted label.
                        predictedLabel = torch.argmax(prediction, 1)
                        # Checking the prediction.
                        if predictedLabel.item() == label[j].item():
                            print("Prediction Success!!!")
                        else:
                            print("Prediction Fail!!!")
                        # Getting the transformation methods.
                        transform = transforms.Compose([transforms.ToPILImage()])
                        # Getting the image data.
                        image = data[j]
                        # Getting the mean and std.
                        mean = [0.485, 0.456, 0.406]
                        std = [0.229, 0.224, 0.225]
                        # Recovering the image.
                        image[0] = image[0] * std[0] + mean[0]
                        image[1] = image[1] * std[1] + mean[1]
                        image[2] = image[2] * std[2] + mean[2]
                        # Getting the image.
                        image = transform(image)
                        # Showing the image.
                        plt.figure(labels[predictedLabel.item()])
                        plt.imshow(image)
                        plt.show()
                        # Getting the command.
                        cmd = input("'Exit' for quiting the prediction: ")
                        if cmd == 'Exit':
                            break
                    if cmd == 'Exit':
                        break
            except:
                print("There are not any trained model, please train one first!!!")
                cmd = 'T'
        else:
            cmd = input("Invalid Input! Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")