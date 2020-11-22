#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/22
#   File Name:      Trainer.py
#   Description:    This file is used to training the model.
#============================================================================================#

# Importing the necessary library.
import matplotlib.pyplot as plt
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from visdom import Visdom
from Utils.Config import argParse
from Utils.DataPreprocessor import dataLoader
from Model.SimpleGANModel import Generator, Discriminator

# Setting the configurator.
Cfg = argParse()

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

# Setting the training function.
def trainer(trainSet):
    # Creating the visdom.
    lossViz = Visdom(env = 'SimpleGANModel')
    # Setting the graph.
    lossGraph = lossViz.line(X = [0], Y = [0], opts = dict(legend = ['GeneratorLoss', 'DiscriminatorLoss'], xlabel = 'Epoches', ylabel = 'Loss', title = 'Generator and Discriminator Loss'), name = 'GeneratorLoss')
    lossViz.line(X = [0], Y = [0], win = lossGraph, update = 'append', name = 'DiscriminatorLoss')
    # Setting the logging file.
    logging.basicConfig(filename = Cfg.logDir + '/logging.txt', filemode = 'a', format = '%(asctime)s %(levelname)s %(message)s', datefmt = '%Y-%D %H:%M:%S %p', level = logging.INFO)
    # Logging the information.
    logging.info(f'''
        Generator Learning Rate:        {Cfg.lrG}
        Discriminator Learning Rate:    {Cfg.lrD}
        Latent Size:                    {Cfg.lt}
        Image Length:                   {Cfg.im}
        Image Size:                     {Cfg.imageSize}
        Epoches:                        {Cfg.epoches}
        Batch Size:                     {Cfg.bs}
        Random Sedd:                    {Cfg.seed}
        GPU ID:                         {Cfg.GPUID}
        Model Directory:                {Cfg.modelDir}
        Logging Directory:              {Cfg.logDir}
        Dataset Directory:              {Cfg.dataDir}    
    ''')
    # Creating the models.
    G = Generator(Cfg.lt, Cfg.imageSize).to(device)
    D = Discriminator(Cfg.imageSize).to(device)
    # Creating the optimizer.
    optimG = optim.Adam(G.parameters(), lr = Cfg.lrG)
    optimD = optim.Adam(D.parameters(), lr = Cfg.lrD)
    # Creating the loss function.
    loss = nn.BCELoss()
    # Initializing the label.
    trueLabel = torch.ones(Cfg.bs, 1).to(device)
    fakeLabel = torch.zeros(Cfg.bs, 1).to(device)
    # Initializing the train loss.
    trainLossesG = []
    trainLossesD = []
    # Training the model.
    for epoch in range(Cfg.epoches):
        # Initializing the training cost and accuracy,
        trainLossD = []
        trainLossG = []
        # Setting the loading bar.
        with tqdm(total = len(trainSet), desc = f'Epoch {epoch + 1}/{Cfg.epoches}', unit = 'batches', ncols = 100) as pbars:
            # Training the model.
            for i, (data, _) in enumerate(trainSet):
                # Initializting the latent space.
                latentSpace = torch.randn(Cfg.bs, Cfg.lt).to(device)
                # Sending the data into corresponding device.
                data = data.reshape(data.shape[0], Cfg.imageSize).to(device)
                # Getting the fake image.
                fakeImage = G(latentSpace)
                # Training the discriminator.
                # Computing the prediction.
                truePrediction = D(data)
                fakePrediction = D(fakeImage.detach())
                # Computing the loss.
                trueCost = loss(truePrediction, trueLabel)
                fakeCost = loss(fakePrediction, fakeLabel)
                costD = trueCost + fakeCost
                # Storing the loss.
                trainLossD.append(costD.item())
                # Clearning the previous gradient descent.
                optimD.zero_grad()
                # Computing the backward.
                costD.backward()
                # Updating the parameters.
                optimD.step()

                # Training the generator.
                # Initializting the latent space.
                latentSpace = torch.randn(Cfg.bs, Cfg.lt).to(device)
                # Getting the fake image.
                fakeImage = G(latentSpace)
                # Computing the prediction.
                fakePrediction = D(fakeImage)
                # Computing the loss.
                costG = loss(fakePrediction, trueLabel)
                # Storing the loss
                trainLossG.append(costG.item())
                # Clearing the previous gradient descent.
                optimD.zero_grad()
                optimG.zero_grad()
                # Computing the backward.
                costG.backward()
                # Updating the parameters.
                optimG.step()

                # Updating the loading bar.
                pbars.update(1)
        # Closing the pbars.
        pbars.close()
        # Logging the data.
        logging.info('Epoch [%d/%d]: G Loss [%.4f] || D Loss [%.4f]' % (epoch + 1, Cfg.epoches, np.sum(trainLossG) / len(trainLossG), np.sum(trainLossD) / len(trainLossD)))
        # Saving the training info.
        trainLossesG.append(np.sum(trainLossG) / len(trainLossG))
        trainLossesD.append(np.sum(trainLossD) / len(trainLossD))
        # Drawing the graph.
        lossViz.line(
            Y = trainLossesG,
            X = [k for k in range(1, len(trainLossesG) + 1)],
            win = lossGraph,
            update = 'new',
            name = 'GeneratorLoss'
        )
        lossViz.line(
            Y = trainLossesD,
            X = [k for k in range(1, len(trainLossesD) + 1)],
            win = lossGraph,
            update = 'new',
            name = 'DiscriminatorLoss'
        )
        # Saving the model.
        torch.save(G.train().state_dict(), Cfg.modelDir + '/SimpleGANGenerator.pt')
    # Saving the graph.
    vis.save(envs = ['SimpleGANModel'])

# Training the model.
if __name__ == "__main__":
    # Getting the training data.
    trainSet = dataLoader.MNIST(Cfg.dataDir, Cfg.bs)
    # Getting the data.
    for i, (data, _) in enumerate(trainSet):
        # Reading the data.
        for j in range(len(data)):
            # Setting the transformating.
            transform = transforms.ToPILImage()
            # Getting the image.
            image = transform(data[j])
            # Plotting the image.
            plt.title("Real MNIST Image")
            plt.imshow(image, cmap = plt.cm.gray)
            plt.show()
            # Getting the command.
            cmd = input("'Exit' for quitting: ")
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
            trainer(trainSet)
            # Getting the command.
            cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
        elif cmd == 'E':
            try:
                # Loading the model.
                model = Generator(Cfg.lt, Cfg.imageSize)
                model.load_state_dict(torch.load(Cfg.modelDir + '/SimpleGANGenerator.pt'))
                # Sending the model into the corresponding device.
                model = model.to(device).eval()
                # Creating the latent space.
                latentSpace = torch.randn(Cfg.bs, Cfg.lt).to(device)
                # Getting the fake image.
                fakeImages = model(latentSpace).reshape((Cfg.bs, 1, 28, 28)).to('cpu')
                # Plotting the image.
                while True:
                    # Getting the image number.
                    i = input("Please input a image number (%d <= number <= %d and 'Exit' for quit): " % (1, Cfg.bs))
                    # Indicate the input.
                    if i == 'Exit':
                        break
                    else:
                        # Indicate the input value.
                        try:
                            i = eval(i)
                            # Setting the transformation.
                            transform = transforms.ToPILImage()
                            # Getting the image.
                            image = transform(fakeImages[i-1])
                            # Plotting the image.
                            plt.title("Generated MNIST Image")
                            plt.imshow(image, cmap = plt.cm.gray)
                            plt.show()
                        except:
                            print("Please input a valid number!!!")
                # Getting the command.
                cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
            except:
                print("There are not any model, please train one first!!!")
                cmd = 'T'
        else:
            cmd = input("Invalid Input! Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")