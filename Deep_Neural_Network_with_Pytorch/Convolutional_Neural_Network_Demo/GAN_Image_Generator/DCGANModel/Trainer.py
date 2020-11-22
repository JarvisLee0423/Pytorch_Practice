#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/21
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
from Model.DCGANModel import Generator, Discrimitor

# Getting the configurator.
Cfg = argParse()

# Getting the device configuration.
if torch.cuda.is_available():
    # Getting the device.
    device = 'cuda'
    # Fixing the GPU device.
    if Cfg.GPUID != -1:
        torch.cuda.set_device(Cfg.GPUID)
    # Fxing the random seed.
    np.random.seed(Cfg.seed)
    torch.cuda.manual_seed(Cfg.seed)
else:
    # Getting the device.
    device = 'cpu'
    # Fixing the random seed.
    np.random.seed(Cfg.seed)
    torch.manual_seed(Cfg.seed)

# Setting the training function.
def trainer(trainSet):
    # Getting the visdom.
    vis = Visdom(env = 'DCGANModel')
    # Setting the loss visdom window.
    lossGraph = vis.line(Y = [0], X = [0], opts = dict(legend = ['GeneratorLoss', 'DiscriminatorLoss'], xlabel = 'Epoches', ylabel = 'Loss', title = 'Generator and Discriminator Losses'), name = 'GeneratorLoss')
    vis.line(Y = [0], X = [0], win = lossGraph, update = 'append', name = 'DiscriminatorLoss')
    # Setting the logging file.
    logging.basicConfig(filename = Cfg.logDir + '/logging.txt', filemode = 'a', level = logging.INFO, format = '%(asctime)s %(levelname)s %(message)s', datefmt = '%Y-%D %H-%M-%S %p')
    # Logging the configuration information.
    logging.info(f'''
        Generator Learning Rate:        {Cfg.lrG}
        Discriminator Learning Rate:    {Cfg.lrD}
        Adam Beta One:                  {Cfg.beta1}
        Adam Beta Two:                  {Cfg.beta2}
        Batch Size:                     {Cfg.bs}
        Latent Size:                    {Cfg.lt}
        Epoches:                        {Cfg.epoches}
        GPU ID:                         {Cfg.GPUID}
        Random Seed:                    {Cfg.seed}
        Model Directory:                {Cfg.modelDir}
        Log Directory:                  {Cfg.logDir}
        Dataset Directory:              {Cfg.dataDir}
    ''')
    # Getting the model.
    G = Generator(Cfg.lt).to(device)
    D = Discrimitor().to(device)
    # Reinitializing the generator's weight and bias.
    G.apply(Generator.weightInit)
    # Setting the loss function.
    loss = nn.BCELoss()
    # Setting the optimizer.
    optimG = optim.Adam(G.parameters(), lr = Cfg.lrG, betas = (Cfg.beta1, Cfg.beta2))
    optimD = optim.Adam(D.parameters(), lr = Cfg.lrD, betas = (Cfg.beta1, Cfg.beta2))
    # Setting the truth label and fake label.
    trueLabel = torch.ones(Cfg.bs, 1).to(device)
    fakeLabel = torch.zeros(Cfg.bs, 1).to(device)
    # Getting the epoch loss.
    trainLossesG = []
    trainLossesD = []
    # Training the model.
    for epoch in range(Cfg.epoches):
        # Getting each iterations' loss.
        trainLossG = []
        trainLossD = []
        # Setting the loading bar.
        with tqdm(total = len(trainSet), desc = f'Epoch {epoch + 1}/{Cfg.epoches}', unit = 'batches', ncols = 100) as pbars:
            for i, (data, _) in enumerate(trainSet):
                # Sending the data into the corresponding device.
                data = data.to(device)
                # Creating the latent space.
                latentSpace = torch.randn(Cfg.bs, Cfg.lt, 1, 1).to(device)
                # Creating fake images.
                fakeImages = G(latentSpace)
                # Training the discriminator.
                truePrediction = D(data)
                fakePrediction = D(fakeImages.detach())
                # Getting the loss.
                costTrue = loss(truePrediction, trueLabel)
                costFake = loss(fakePrediction, fakeLabel)
                costD = costTrue + costFake
                # Storing the loss of discriminator.
                trainLossD.append(costD.item())
                # Clearing the gradients.
                optimD.zero_grad()
                costD.backward()
                optimD.step()

                # Creating the latent space.
                latentSpace = torch.randn(Cfg.bs, Cfg.lt, 1, 1).to(device)
                # Creating fake images.
                fakeImages = G(latentSpace)
                # Training the generator.
                generatedPrediction = D(fakeImages)
                # Getting the loss.
                costG = loss(generatedPrediction, trueLabel)
                # Storing the loss of generator.
                trainLossG.append(costG.item())
                # Clearing the gradients.
                optimD.zero_grad()
                optimG.zero_grad()
                costG.backward()
                optimG.step()

                # Updating the loading bar.
                pbars.update(1)
        # Closing the loading bar.
        pbars.close()
        # Getting the epoch loss.
        trainLossesD.append(np.sum(trainLossD) / len(trainLossD))
        trainLossesG.append(np.sum(trainLossG) / len(trainLossG))
        # Logging the losses.
        logging.info('Epoch [%d/%d]: G Loss [%.4f] || D Loss [%.4f]' % (epoch + 1, Cfg.epoches, np.sum(trainLossG) / len(trainLossG), np.sum(trainLossD) / len(trainLossD)))
        # Drawing the losses.
        vis.line(
            X = [k for k in range(1, len(trainLossesD) + 1)],
            Y = trainLossesD,
            win = lossGraph,
            update = 'new',
            name = 'DiscriminatorLoss'
        )
        vis.line(
            X = [k for k in range(1, len(trainLossesG) + 1)],
            Y = trainLossesG,
            win = lossGraph,
            update = 'new',
            name = 'GeneratorLoss'
        )
        # Saving the model.
        torch.save(G.train().state_dict(), Cfg.modelDir + '/DCGANGenerator.pt')

# Training the model.
if __name__ == "__main__":
    # Getting the data.
    trainSet = dataLoader.CELEBA(Cfg.dataDir, Cfg.bs)
    # Outputing the data.
    for _, (images, _) in enumerate(trainSet):
        # Setting the transformation.
        transform = transforms.Compose([
            transforms.Normalize(
                mean = (-1, -1, -1),
                std = (2, 2, 2)
            ),
            transforms.ToPILImage()
        ])
        # Getting the image.
        for i in range(len(images)):
            # Transforming the image.
            image = transform(images[i])
            # Drawing the image.
            plt.title("Real Face Image")
            plt.imshow(image)
            plt.show()
            # Checking whether continuing to showing the image.
            cmd = input("'Exit' for quit: ")
            if cmd == 'Exit':
                break
        # Checking whether continuing to showing the image.
        if cmd == 'Exit':
            break
    #Getting the command.
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
                model = Generator(Cfg.lt)
                model.load_state_dict(torch.load(Cfg.modelDir + '/DCGANGenerator.pt'))
                # Sending the model into the corresponding device.
                model = model.to(device).eval()
                # Creating the latent space.
                latentSpace = torch.randn(Cfg.bs, Cfg.lt, 1, 1).to(device)
                # Getting the fake image.
                fakeImages = model(latentSpace).to('cpu')
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
                            transform = transforms.Compose([
                                transforms.Normalize(
                                    mean = (-1, -1, -1),
                                    std = (2, 2, 2)
                                ),
                                transforms.ToPILImage()
                            ])
                            # Getting the image.
                            image = transform(fakeImages[i-1])
                            # Plotting the image.
                            plt.title("Generated Face Image")
                            plt.imshow(image)
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