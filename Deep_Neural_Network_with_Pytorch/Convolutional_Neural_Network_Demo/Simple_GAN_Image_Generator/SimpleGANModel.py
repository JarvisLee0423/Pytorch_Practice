#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/08/28
#   Project Name:       SimpleGANModel.py
#   Description:        Build a simple GAN model to generating the image in MNIST datastes.
#   Model Description:  Input               ->  MNIST Images and Fake Images
#                       Generator           ->  Conv2d      ->  Kernel Size:    (3, 3)
#                                                           ->  Channel:        10 
#                                                           ->  Same Padding
#                                           ->  MaxPool2d   ->  Kernel Size:    (2, 2)
#                                                           ->  Stride:         2
#                                           ->  ReLu
#                                           ->  Conv2d      ->  Kernel Size:    (1, 1)
#                                                           ->  Channel:        5
#                                           ->  ReLu
#                                           ->  Linear      ->  Linear(980, 784)
#                       Discriminator       ->  Conv2d      ->  Kernel Size:    (3, 3)
#                                                           ->  Channel:        5
#                                                           ->  Same Padding
#                                           ->  MaxPool2d   ->  Kernel Size:    (2, 2)
#                                                           ->  Stride:         2
#                                           ->  ReLu
#                                           ->  Conv2d      ->  Kernel Size:    (5, 5)
#                                                           ->  Channel Size:   10
#                                                           ->  Same Padding
#                                           ->  MaxPool2d   ->  Kernel Size:    (2, 2)
#                                                           ->  Stride:         2
#                                           ->  ReLu
#                                           ->  Linear      ->  Linear(490, 200)
#                                           ->  Linear      ->  Linear(200, 1)      
#============================================================================================#

# Importing the necessary library.
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

# Fixing the computer device and random size.
if torch.cuda.is_available():
    # Fixing the computer device.
    torch.cuda.set_device(0)
    # Fixing the random size.
    #torch.cuda.manual_seed(1)
    # Setting the computer device.
    device = 'cuda'
else:
    # Fixing the random seed.
    #torch.manual_seed(1)
    # Setting the computer device.
    device = 'cpu'

# Setting the hyperparameters.
# The value of the learning rate.
learningRate = 2e-4
# The value of the batch size.
batchSize = 32
# The value of the epoches.
epoches = 30

# Creating the data loader.
class dataLoader():
    # Defining the MNIST data loader.
    @staticmethod
    def MNIST(batchSize):
        # Checking whether download the data.
        if os.path.exists('./Datasets/MNIST/'):
            download = False
        else:
            download = True
        # Getting the training data.
        trainData = datasets.MNIST(
            root = './Datasets/',
            train = True,
            transform = transforms.ToTensor(),
            download = download
        )
        # Getting the training set.
        trainSet = DataLoader(
            trainData,
            batch_size = batchSize,
            shuffle = True
        )
        # Returning the training sets.
        return trainSet

# Creating the generator.
class Generator(nn.Module):
    # Creating the constructor.
    def __init__(self):
        # Inheriting the super constructor.
        super(Generator, self).__init__()
        # Defining the model.
        self.conv1 = nn.Conv2d(1, 10, (3, 3), padding = 1)
        self.pool1 = nn.MaxPool2d((2, 2), 2)
        self.conv2 = nn.Conv2d(10, 5, (1, 1))
        self.linear = nn.Linear(980, 784)
    # Defining the forward.
    def forward(self, x):
        # [batchSize, 1, 28, 28] -> [batchSize, 10, 28, 28]
        x = self.conv1(x)
        # [batchSize, 10, 28, 28] -> [batchSize, 10, 14, 14]
        x = self.pool1(x)
        x = F.relu(x)
        # [batchSize, 10, 14, 14] -> [batchSize, 5, 14, 14]
        x = self.conv2(x)
        x = F.relu(x)
        # [batchSize, 5, 14, 14] -> [batchSize, 980]
        x = x.reshape(-1, 980)
        # [batchSize, 980] -> [batchSize, 784]
        x = self.linear(x)
        # Returning the result.
        return torch.tanh(x)

# Creating the discriminator.
class Discriminator(nn.Module):
    # Creating the constructor.
    def __init__(self):
        # Inhertiting the super constructor.
        super(Discriminator, self).__init__()
        # Defining the model.
        self.conv1 = nn.Conv2d(1, 5, (3, 3), padding = 1)
        self.pool1 = nn.MaxPool2d((2, 2), 2)
        self.conv2 = nn.Conv2d(5, 10, (5, 5), padding = 2)
        self.pool2 = nn.MaxPool2d((2, 2), 2)
        self.linear1 = nn.Linear(490, 200)
        self.linear2 = nn.Linear(200, 1)
    # Defining the forward.
    def forward(self, x):
        # [batchSize, 1, 28, 28] -> [batchSize, 5, 28, 28]
        x = self.conv1(x)
        # [batchSize, 5, 28, 28] -> [batchSize, 5, 14, 14]
        x = self.pool1(x)
        x = F.relu(x)
        # [batchSize, 5, 14, 14] -> [batchSize, 10, 14, 14]
        x = self.conv2(x)
        # [batchSize, 10, 14, 14] -> [batchSize, 10, 7, 7]
        x = self.pool2(x)
        x = F.relu(x)
        # [batchSize, 10, 7, 7] -> [batchSize, 490]
        x = x.reshape(-1, 490)
        # [batchSize, 490] -> [batchSize, 200]
        x = self.linear1(x)
        # [batchSize, 200] -> [batchSize, 1]
        x = self.linear2(x)
        # Returning the result.
        return torch.sigmoid(x)

# Creating the model.
class SimpleGANModelNN():
    # Training the model.
    @staticmethod
    def trainer(trainSet, epoches, batchSize, learningRate):
        # Creating the models.
        G = Generator().to(device)
        D = Discriminator().to(device)
        # Creating the optimizer.
        optimG = optim.Adam(G.parameters(), lr = learningRate)
        optimD = optim.Adam(D.parameters(), lr = learningRate)
        # Creating the loss function.
        loss = nn.BCELoss()
        # Initializing the label.
        trueLabel = torch.ones([batchSize, 1]).to(device)
        fakeLabel = torch.zeros([batchSize, 1]).to(device)
        # Initializing the train loss.
        trainLossesG = 0
        trainLossesD = 0
        # Training the model.
        for epoch in range(epoches):
            # Initializing the training cost and accuracy,
            trainLossD = []
            trainLossG = []
            # Training the model.
            for i, (data, _) in enumerate(trainSet):
                # Sending the data into corresponding device.
                data = data.to(device)
                # Initializting the latent space.
                latentSpace = torch.rand_like(data).to(device)
                # Getting the fake image.
                fakeImage = G(latentSpace).reshape((data.shape[0], 1, 28, 28))
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
                # Computing the prediction.
                fakePrediction = D(fakeImage)
                # Computing the loss.
                costG= loss(fakePrediction, trueLabel)
                # Storing the loss
                trainLossG.append(costG.item())
                # Clearing the previous gradient descent.
                optimD.zero_grad()
                optimG.zero_grad()
                # Computing the backward.
                costG.backward()
                # Updating the parameters.
                optimG.step()
            # Selecting the best model.
            if (trainLossesG == 0 and trainLossesD == 0) or ((np.sum(trainLossG) / len(trainLossG)) < trainLossesG or (np.sum(trainLossD) / len(trainLossD)) < trainLossesD):
                # Saving the model.
                torch.save(G.train().state_dict(), './Deep_Neural_Network_with_Pytorch/Convolutional_Neural_Network_Demo/Simple_GAN_Image_Generator/SimpleGANModel.pt')
                print("Generator Saved")
            # Storing the losses.
            trainLossesG = (np.sum(trainLossG) / len(trainLossG))
            trainLossesD = (np.sum(trainLossD) / len(trainLossD))
            # Printing the training information.
            print("The epoch " + str(epoch + 1) + "/" + str(epoches) + " training: D Loss = " + str(np.sum(trainLossD) / len(trainLossD)) + " || G Loss = " + str(np.sum(trainLossG) / len(trainLossG)))

# Training the model.
if __name__ == "__main__":
    pass
    # # Getting the training data.
    # trainSet = dataLoader.MNIST(batchSize)
    # # Getting the data.
    # for i, (data, _) in enumerate(trainSet):
    #     # Reading the data.
    #     for j in range(len(data)):
    #         # Setting the transformating.
    #         transform = transforms.ToPILImage()
    #         # Getting the image.
    #         image = transform(data[j])
    #         # Plotting the image.
    #         plt.imshow(image, cmap = plt.cm.gray)
    #         plt.show()
    #         # Getting the command.
    #         cmd = input("'Exit' for quitting: ")
    #         # Handling the command.
    #         if cmd == 'Exit':
    #             break
    #     if cmd == 'Exit':
    #         break
    # # Getting the command.
    # cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
    # # Handling the command.
    # while cmd != 'Exit':
    #     if cmd == 'T':
    #         # Training the model.
    #         SimpleGANModelNN.trainer(trainSet, epoches, batchSize, learningRate)
    #         # Getting the command.
    #         cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
    #     elif cmd == 'E':
    #         try:
    #             # Loading the model.
    #             model = Generator()
    #             model.load_state_dict(torch.load('./Deep_Neural_Network_with_Pytorch/Convolutional_Neural_Network_Demo/Simple_GAN_Image_Generator/SimpleGANModel.pt'))
    #             # Sending the model into the corresponding device.
    #             model = model.to(device).eval()
    #             # Creating the latent space.
    #             for _, (data, _) in enumerate(trainSet):
    #                 # Initializting the latent space.
    #                 latentSpace = torch.rand_like(data).to(device)
    #                 break
    #             # Getting the fake image.
    #             fakeImage = model(latentSpace).reshape((batchSize, 1, 28, 28)).to('cpu')
    #             # Plotting the image.
    #             for i in range(batchSize):
    #                 # Setting the transformation.
    #                 transform = transforms.ToPILImage()
    #                 # Getting the image.
    #                 image = transform(fakeImage[i])
    #                 # Plotting the image.
    #                 plt.imshow(image, cmap = plt.cm.gray)
    #                 plt.show()
    #             # Getting the command.
    #             cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
    #         except:
    #             print("There are not any model, please train one first!!!")
    #             cmd = 'T'
    #     else:
    #         cmd = input("Invalid Input! Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")