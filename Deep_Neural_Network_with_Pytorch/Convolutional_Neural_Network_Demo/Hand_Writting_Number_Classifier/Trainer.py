#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/22
#   File Name:      Trainer.py
#   Description:    This file is used to training the model.
#============================================================================================#

# Importing the necessary library.
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from visdom import Visdom
from Utils.Config import argParse
from Utils.DataPreprocessor import dataLoader
from Model.HandWrittingNumberModel import HandWrittingNumberModelNN

# Getting the configurator.
Cfg = argParse()

# Creating the directory.
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
    # Fixing the device.
    np.random.seed(Cfg.seed)
    torch.manual_seed(Cfg.seed)

# Defining the evaluating model.
def evaluator(model, loss, devSet):
    # Initializing the evaluating loss and accuracy.
    evalLoss = []
    evalAcc = []
    # Evaluating the model.
    for i, (data, label) in enumerate(devSet):
        # Sending the data into corresponding device.
        data = data.to(device)
        label = label.to(device)
        # Evaluating the model.
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
    # Returning the evaluating loss and accuracy.
    return np.sum(evalLoss) / len(evalLoss), np.sum(evalAcc) / len(evalAcc)

# Defining the training method.
def trainer(trainSet, devSet):
    # Setting the visdom.
    vis = Visdom(env = 'HandWrittingNumberClassifierModel')
    # Creating the graphs.
    lossGraph = vis.line(X = [0], Y = [0], opts = dict(legend = ['TrainingLoss', 'EvaluatingLoss'], xlabel = 'Epoches', ylabel = 'Loss', title = 'Training and Evaluating Loss'), name = 'TrainingLoss')
    vis.line(X = [0], Y = [0], win = lossGraph, update = 'append', name = 'EvaluatingLoss')
    accGraph = vis.line(X = [0], Y = [0], opts = dict(legend = ['TrainingAcc', 'EvaluatingAcc'], xlabel = 'Epoches', ylabel = 'Acc', title = 'Training and Evaluating Acc'), name = 'TrainingAcc')
    vis.line(X = [0], Y = [0], win = accGraph, update = 'append', name = 'EvaluatingAcc')
    # Creating the logging.
    logging.basicConfig(filename = Cfg.logDir + '/logging.txt', filemode = 'a', level = logging.INFO, format = '%(asctime)s %(levelname)s %(message)s', datefmt = '%Y-%D %H:%M:%S %p')
    # Logging the information.
    logging.info(f'''
        Learning Rate:          {Cfg.lr}
        Batch Size:             {Cfg.bs}
        Epoches:                {Cfg.epoches}
        Random Seed:            {Cfg.seed}
        GPU ID:                 {Cfg.GPUID}
        Model Directory:        {Cfg.modelDir}
        Log Directory:          {Cfg.logDir}
        Dataset Directory:      {Cfg.dataDir}
        Test Data Directory:    {Cfg.testDir}
    ''')
    # Creating the model.
    model = HandWrittingNumberModelNN(inChannel = 1, classSize = 10)
    # Sending the model into corresponding device.
    model = model.to(device)
    # Setting the optimizer.
    optimizer = optim.Adam(model.parameters(), lr = Cfg.lr)
    # Setting the loss function.
    loss = nn.CrossEntropyLoss()
    # Initializing the training accuracy.
    trainAccs = []
    # Initializing the training loss.
    trainLosses = []
    # Initializing the evaluating accuracy.
    evalAccs = []
    # Initializing the evaluating loss.
    evalLosses = []
    # Training the model.
    for epoch in range(Cfg.epoches):
        # Initializing the training loss and accuracy.
        trainLoss = []
        trainAcc = []
        # Setting the loading bar.
        with tqdm(total = len(trainSet), desc = f'Epoch {epoch + 1}/{Cfg.epoches}', unit = 'batches', ncols = 100) as pbars:
            # Getting the data.
            for i, (data, label) in enumerate(trainSet):
                # Sending the data into the corresponding device.
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
                # Computing the backward propagation.
                cost.backward()
                # Updating the parameters.
                optimizer.step()
                # Computing the accuracy.
                accuracy = (torch.argmax(prediction, 1) == label)
                accuracy = accuracy.sum().float() / len(accuracy)
                # Storing the loss.
                trainLoss.append(cost.item())
                # Storing the accuracy.
                trainAcc.append(accuracy.item())

                # Updating the loading bar.
                pbars.update(1)
        # Closing the loading bar.
        pbars.close()
        # Printing the information.
        print('Evaluating...')
        # Evaluating the model.
        evalLoss, evalAcc = evaluator(model.eval(), loss, devSet)
        # Storing the training loss and acc.
        trainLosses.append(np.sum(trainLoss) / len(trainLoss))
        trainAccs.append(np.sum(trainAcc) / len(trainAcc))
        # Storing the evaluating loss and acc.
        evalLosses.append(evalLoss)
        evalAccs.append(evalAcc)
        # Updating the graph.
        vis.line(
            Y = trainLosses,
            X = [k for k in range(1, len(trainLosses) + 1)],
            win = lossGraph,
            update = 'new',
            name = 'TrainingLoss'
        )
        vis.line(
            Y = evalLosses,
            X = [k for k in range(1, len(evalLosses) + 1)],
            win = lossGraph,
            update = 'new',
            name = 'EvaluatingLoss'
        )
        vis.line(
            Y = trainAccs,
            X = [k for k in range(1, len(trainAccs) + 1)],
            win = accGraph,
            update = 'new',
            name = 'TrainingAcc'
        )
        vis.line(
            Y = evalAccs,
            X = [k for k in range(1, len(evalAccs) + 1)],
            win = accGraph,
            update = 'new',
            name = 'EvaluatingAcc'
        )
        # Logging the training and evaluating information.
        logging.info('Epoch [%d/%d] -> Training: Loss [%.4f] - Acc [%.4f] || Evaluating: Loss [%.4f] - Acc [%.4f]' % (epoch + 1, Cfg.epoches, np.sum(trainLoss) / len(trainLoss), np.sum(trainAcc) / len(trainAcc), evalLoss, evalAcc))
        # Saving the best model.
        torch.save(model.train().state_dict(), Cfg.modelDir + f'/HandWrittingNumberModel-Epoch{epoch + 1}.pt')
        logging.info("Model Saved")
        # Converting the model mode.
        model.train()

# Training the model.
if __name__ == "__main__":
    # Generating the training and development sets.
    trainSet, devSet = dataLoader.MNIST(Cfg.dataDir, Cfg.bs)
    # Outputing the data as image.
    for _, (data, _) in enumerate(trainSet):
        # Setting the transform method.
        transform = transforms.ToPILImage()
        # Transforming the tensor data into image.
        for j in range(Cfg.bs):
            image = transform(data[j])
            # Plotting the image.
            plt.imshow(image, cmap = plt.cm.gray)
            # Drawing the image.
            plt.show()
            # Getting the cmd.
            cmd = input("'Exit' for quiting the data displaying: ")
            # Handling the cmd.
            if cmd == "Exit":
                break
            else:
                continue
        # Handling the cmd.
        if cmd == "Exit":
            break
        else:
            continue
    # Getting the command.
    cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
    # Handling the command.
    while cmd != 'Exit':
        if cmd == 'T':
            # Training the model.
            trainer(trainSet, devSet)
            # Getting the command.
            cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
        elif cmd == 'E':
            try:
                # Creating the model.
                model = HandWrittingNumberModelNN(1, 10)
                # Loading the model.
                model.load_state_dict(torch.load(Cfg.modelDir + f'/HandWrittingNumberModel-Epoch{Cfg.epoches}.pt'))
                # Sending the model into the corresponding device.
                model = model.to(device)
                # Converting the model mode.
                model.eval()
                # Testing the model.
                # Getting the root of the testing model.
                root = Cfg.testDir
                # Getting the testing data.
                for filename in os.listdir(root):
                    # Getting the image.
                    image = Image.open(os.path.join(root, filename))
                    # Setting the transformation.
                    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
                    # Getting the testing data. [1, 28, 28] -> [1, 1, 28, 28]
                    inputData = transform(image.convert("1")).unsqueeze(0)
                    # Sending the data into the corresponding device.
                    inputData = inputData.to(device)
                    # Testing the data.
                    prediction = model(inputData)
                    # Getting the predicted label.
                    prediction = int(torch.argmax(prediction, 1))
                    # Printing the predicted label.
                    print("The predicted label of " + filename + " is: " + str(prediction))
                    #Drawing the testing image.
                    plt.imshow(image)
                    plt.show()
                    # Getting the command.
                    cmd = input("Input 'Exit' to stop the testing: ")
                    # Handling the command.
                    if cmd == 'Exit':
                        break
                    else:
                        continue
                # Giving the hint.
                print("Testing completed!!!")
            except:
                # Giving the hint.
                print("There are not any trained model, please training one first!!!")
                cmd = 'T'
        else:
            cmd = input("Invalid Input! Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")