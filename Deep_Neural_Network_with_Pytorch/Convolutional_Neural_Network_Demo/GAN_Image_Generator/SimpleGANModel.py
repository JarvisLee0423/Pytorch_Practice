#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/09/20
#   Project Name:       SimpleGANModel.py
#   Description:        Build a simple GAN model to generating the image in MNIST datastes.
#   Model Description:  Input               ->  MNIST Images and Fake Images
#                       Generator           ->  Linaer (latentSize, 200)
#                                           ->  ReLu
#                                           ->  Linear (200, 400)
#                                           ->  ReLu
#                                           ->  Linear (400, imageSize)
#                                           ->  tanh
#                       Discriminator       ->  Linear (imageSize, 400)
#                                           ->  LeakyReLu (0.2)
#                                           ->  Linear (400, 200)
#                                           ->  LeakyReLu (0.2)
#                                           ->  Linear (200, 1)
#                                           ->  Sigmoid    
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
learningRateG = 2e-4
learningRateD = 2e-4
# The value of the batch size.
batchSize = 32
# The value of the latent size.
latentSize = 100
# The value of the image size.
imageSize = 28 * 28
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
        # Setting the transformation method.
        transformMethod = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = (0.5,),
                    std = (0.5,)
                )
            ]
        )
        # Getting the training data.
        trainData = datasets.MNIST(
            root = './Datasets/',
            train = True,
            transform = transformMethod,
            download = download
        )
        # Getting the training set.
        trainSet = DataLoader(
            trainData,
            batch_size = batchSize,
            shuffle = True,
            drop_last = True
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
        self.linear1 = nn.Linear(latentSize, 200)
        self.linear2 = nn.Linear(200, 400)
        self.linear3 = nn.Linear(400, imageSize)
    # Defining the forward.
    def forward(self, x):
        # [batchSize, latentSpace] -> [batchSize, 200]
        x = self.linear1(x)
        x = F.relu(x)
        # [batchSize, 200] -> [batchSize, 400]
        x = self.linear2(x)
        x = F.relu(x)
        # [batchSize, 400] -> [batchSize, imageSize]
        x = self.linear3(x)
        # Returning the result.
        return torch.tanh(x)

# Creating the discriminator.
class Discriminator(nn.Module):
    # Creating the constructor.
    def __init__(self):
        # Inhertiting the super constructor.
        super(Discriminator, self).__init__()
        # Defining the model.
        self.linear1 = nn.Linear(imageSize, 400)
        self.linear2 = nn.Linear(400, 200)
        self.linear3 = nn.Linear(200, 1)
    # Defining the forward.
    def forward(self, x):
        # [batchSize, latentSpace] -> [batchSize, 400]
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.2)
        # [batchSize, 400] -> [batchSize, 200]
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.2)
        # [batchSize, 200] -> [batchSize, 1]
        x = self.linear3(x)
        # Returning the result.
        return torch.sigmoid(x)

# Creating the model.
class SimpleGANModelNN():
    # Training the model.
    @staticmethod
    def trainer(trainSet, epoches, batchSize, learningRateG, learningRateD, latentSize, imageSize):
        # Creating the models.
        G = Generator().to(device)
        D = Discriminator().to(device)
        # Creating the optimizer.
        optimG = optim.Adam(G.parameters(), lr = learningRateG)
        optimD = optim.Adam(D.parameters(), lr = learningRateD)
        # Creating the loss function.
        loss = nn.BCELoss()
        # Initializing the label.
        trueLabel = torch.ones(batchSize, 1).to(device)
        fakeLabel = torch.zeros(batchSize, 1).to(device)
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
                # Initializting the latent space.
                latentSpace = torch.randn(batchSize, latentSize).to(device)
                # Sending the data into corresponding device.
                data = data.reshape(data.shape[0], imageSize).to(device)
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
                latentSpace = torch.randn(batchSize, latentSize).to(device)
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
            # Saving the model.
            torch.save(G.train().state_dict(), './Deep_Neural_Network_with_Pytorch/Convolutional_Neural_Network_Demo/GAN_Image_Generator/SimpleGANGenerator.pt')
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
    #         plt.title("Real MNIST Image")
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
    #         SimpleGANModelNN.trainer(trainSet, epoches, batchSize, learningRateG, learningRateD, latentSize, imageSize)
    #         # Getting the command.
    #         cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
    #     elif cmd == 'E':
    #         try:
    #             # Loading the model.
    #             model = Generator()
    #             model.load_state_dict(torch.load('./Deep_Neural_Network_with_Pytorch/Convolutional_Neural_Network_Demo/GAN_Image_Generator/SimpleGANGenerator.pt'))
    #             # Sending the model into the corresponding device.
    #             model = model.to(device).eval()
    #             # Creating the latent space.
    #             latentSpace = torch.randn(batchSize, latentSize).to(device)
    #             # Getting the fake image.
    #             fakeImages = model(latentSpace).reshape((batchSize, 1, 28, 28)).to('cpu')
    #             # Plotting the image.
    #             while True:
    #                 # Getting the image number.
    #                 i = input("Please input a image number (%d <= number <= %d and 'Exit' for quit): " % (1, batchSize))
    #                 # Indicate the input.
    #                 if i == 'Exit':
    #                     break
    #                 else:
    #                     # Indicate the input value.
    #                     try:
    #                         i = eval(i)
    #                         # Setting the transformation.
    #                         transform = transforms.ToPILImage()
    #                         # Getting the image.
    #                         image = transform(fakeImages[i-1])
    #                         # Plotting the image.
    #                         plt.title("Generated MNIST Image")
    #                         plt.imshow(image, cmap = plt.cm.gray)
    #                         plt.show()
    #                     except:
    #                         print("Please input a valid number!!!")
    #             # Getting the command.
    #             cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
    #         except:
    #             print("There are not any model, please train one first!!!")
    #             cmd = 'T'
    #     else:
    #         cmd = input("Invalid Input! Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")