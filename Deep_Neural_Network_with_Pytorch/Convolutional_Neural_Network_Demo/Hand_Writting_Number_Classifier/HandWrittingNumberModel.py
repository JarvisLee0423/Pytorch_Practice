#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/08/24
#   Project Name:       HandWrittingNumberModel.py
#   Description:        Build the CNN model to solve the hand writting number problem.
#   Model Description:  Input               ->  Hand writting number Gray-Level Image
#                       Conv2d Layer        ->  InChannel = 1
#                                           ->  OutChannel = 64
#                                           ->  Kernel Size = (3, 3)
#                                           ->  Padding = 1
#                                           ->  Output Size = 28 * 28 * 64
#                       MaxPool2d Layer     ->  Kernel Size = (2, 2)
#                                           ->  Stride = 2
#                                           ->  Output Size = 14 * 14 * 64
#                       Conv2d Layer        ->  InChannel = 64
#                                           ->  OutChannel = 128
#                                           ->  Kernel Size = (5, 5)
#                                           ->  Output Size = 10 * 10 * 128
#                       MaxPool2d Layer     ->  Kernel Size = (2, 2)
#                                           ->  Stride = 2
#                                           ->  Output Size = 5 * 5 * 128
#                       Conv2d Layer        ->  InChannel = 128
#                                           ->  OutChannel = 64
#                                           ->  Kernel Size = (1, 1)
#                                           ->  Output Size = 5 * 5 * 64
#                       Linear Layer        ->  Shrinking the size of the input
#                       Linear Layer        ->  Computing the prediction
#============================================================================================#

# Importing the necessary library.
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from PIL import Image

# Fixing the computer device and random seed.
# Indicating whether the computer has the GPU or not.
if torch.cuda.is_available:
    # Fixing the computer device.
    torch.cuda.set_device(0)
    # Fixing the random seed.
    torch.cuda.manual_seed(1)
    # Setting the computer device.
    device = 'cuda'
else:
    # Fixing the random seed.
    torch.manual_seed(1)
    # Setting the computer device.
    device = 'cpu'

# Setting the hyperparameters.
# Setting the channel of the input image.
inChannel = 1
# Setting the number of the classes.
classSize = 10
# Setting the value of the learning rate.
learningRate = 1e-3
# Setting the value of the batch size.
batchSize = 128
# Setting the value of the epoches.
epoches = 10

# Creating the data generator.
class dataGenerator():
    # Creating the data generator.
    @staticmethod
    def generator():
        # Indicating whether there are the local dataset or not.
        if os.path.exists('./Datasets/'):
            downloadDataset = False
        else:
            downloadDataset = True
        # Setting the training and development data.
        trainData = datasets.MNIST(
            # Getting the dataset from the root.
            root = './Datasets/',
            # Setting the data mode.
            train = True,
            # Preprocessing the data.
            transform = transforms.ToTensor(),
            # Setting whether to download the data.
            download = downloadDataset
        )
        # Setting the development data.
        devData = datasets.MNIST(
            # Getting the dataset from the root.
            root = './Datasets/',
            # Setting the data mode.
            train = False,
            # Preprocessing the data.
            transform = transforms.ToTensor(),
            # Setting whether to download the data.
            download = downloadDataset
        )
        # Getting the training set and development set.
        trainSet = DataLoader(
            # Getting the training data.
            trainData,
            # Setting the batch size.
            batch_size = batchSize,
            # Indicating whether to shuffle the data.
            shuffle = True  
        )
        devSet = DataLoader(
            # Getting the development data.
            devData,
            # Setting the batch size.
            batch_size = batchSize,
            # Indicating whether to shuffle the data.
            shuffle = False
        )
        # Returning the training and development sets.
        return trainSet, devSet

# Creating the model.
class HandWrittingNumberModelNN(nn.Module):
    # Creating the constructor.
    def __init__(self, inChannel, classSize):
        # Inheriting the super constructor.
        super(HandWrittingNumberModelNN, self).__init__()
        # Setting the first convolutional layer.
        self.conv1 = nn.Conv2d(inChannel, inChannel, (3, 3), padding = 1)
        # Setting the first point-wise convolutional layer.
        self.pointwise1 = nn.Conv2d(inChannel, 64, (1, 1))
        # Setting the first max pooling layer.
        self.maxPool1 = nn.MaxPool2d((2, 2), 2)
        # Setting the second convolutional layer.
        self.conv2 = nn.Conv2d(64, 64, (5, 5))
        # Setting the second point-wise convolutional layer.
        self.pointwise2 = nn.Conv2d(64, 128, (1, 1))
        # Setting the second max pooling layer.
        self.maxPool2 = nn.MaxPool2d((2, 2), 2)
        # Setting the last convolutional layer.
        self.lastconv = nn.Conv2d(128, 64, (1, 1))
        # Setting the first linear layer.
        self.linear_1 = nn.Linear(1600, 400)
        # Computing the prediction.
        self.linear_2 = nn.Linear(400, classSize)
    # Defining the forward propagation.
    def forward(self, x):
        # Applying the first convolutional layer. [batchSize, 1, 28, 28] -> [batchSize, 1, 28, 28]
        x = self.conv1(x)
        # Applying the first point wise convolutional layer. [batchSize, 1, 28, 28] -> [batchSize, 64, 28, 28]
        x = self.pointwise1(x)
        # Applying the first max pooling layer. [batchSize, 64, 28, 28] -> [batchSize, 64, 14, 14]
        x = self.maxPool1(x)
        # Applying the second convolutional layer. [batchSize, 64, 14, 14] -> [batchSize, 64, 10, 10]
        x = self.conv2(x)
        # Applying the second point wise convolutional layer. [batchSize, 64, 10, 10] -> [batchSize, 128, 10, 10]
        x = self.pointwise2(x)
        # Applying the second max pooling layer. [batchSize, 128, 10, 10] -> [batchSize, 128, 5, 5]
        x = self.maxPool2(x)
        # Applying the last convolutional layer. [batchSize, 128, 5, 5] -> [batchSize, 64, 5, 5]
        x = self.lastconv(x)
        # Flattening the data. [batchSize, 64, 5, 5] -> [batchSize, 1600]
        x = x.view(-1, 1600)
        # Applying the first linear layer. [batchSize, 1600] -> [batchSize, 400]
        x = self.linear_1(x)
        # Applying the second linear layer. [batchSize, 400] -> [batchSize, classSize]
        x = self.linear_2(x)
        # Returning the prediction.
        return x
    # Defining the training method.
    @staticmethod
    def trainer(model, optimizer, loss, trainSet, devSet, epoches):
        # Indicating whether the model is correct.
        assert type(model) != type(HandWrittingNumberModelNN)
        # Sending the model into corresponding device.
        model = model.to(device)
        # Initializing the evaluating accuracy.
        evalAccs = []
        # Training the model.
        for epoch in range(epoches):
            # Initializing the training loss and accuracy.
            trainLoss = []
            trainAcc = []
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
                # Storing the accuracy.
                trainAcc.append(accuracy.item())
                # # Printing the training information.
                # if i % 100 == 0:
                #     print("The iteration " + str(i) + " training: Loss = " + str(cost.item()) + " || Acc = " + str(accuracy.item()))
            # Evaluating the model.
            evalLoss, evalAcc = HandWrittingNumberModelNN.evaluator(model.eval(), loss, devSet)
            # Saving the best model.
            if len(evalAccs) == 0 or evalAcc >= max(evalAccs):
                torch.save(model.train().state_dict(), './Deep_Neural_Network_with_Pytorch/Convolutional_Neural_Network_Demo/Hand_Writting_Number_Classifier/HandWrittingNumberModel.pt')
                print("Model Saved")
            # Storing the evaluating accuracy.
            evalAccs.append(evalAcc)
            # Converting the model mode.
            model.train()
            # Printing the training information.
            print("The epoch " + str(epoch + 1) + " training: Loss = " + str(np.sum(trainLoss) / len(trainLoss)) + " || Acc = " + str(np.sum(trainAcc) / len(trainAcc)))
            print("The epoch " + str(epoch + 1) + " evaluating: Loss = " + str(evalLoss) + " || Acc = " + str(evalAcc))
    # Defining the evaluating model.
    @staticmethod
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
    
# Training the model.
if __name__ == "__main__":
    pass
    # # Generating the training and development sets.
    # trainSet, devSet = dataGenerator.generator()
    # # Outputing the data as image.
    # for _, (data, _) in enumerate(trainSet):
    #     # Setting the transform method.
    #     transform = transforms.ToPILImage()
    #     # Transforming the tensor data into image.
    #     for j in range(batchSize):
    #         image = transform(data[j])
    #         # Plotting the image.
    #         plt.imshow(image, cmap = plt.cm.gray)
    #         # Drawing the image.
    #         plt.show()
    #         # Getting the cmd.
    #         cmd = input("'Exit' for quiting the data displaying: ")
    #         # Handling the cmd.
    #         if cmd == "Exit":
    #             break
    #         else:
    #             continue
    #     # Handling the cmd.
    #     if cmd == "Exit":
    #         break
    #     else:
    #         continue
    # # Creating the model.
    # model = HandWrittingNumberModelNN(inChannel, classSize)
    # # Setting the optimizer.
    # optimizer = optim.Adam(model.parameters(), lr = learningRate)
    # # Setting the loss function.
    # loss = nn.CrossEntropyLoss()
    # # Getting the command.
    # cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
    # # Handling the command.
    # while cmd != 'Exit':
    #     if cmd == 'T':
    #         # Training the model.
    #         HandWrittingNumberModelNN.trainer(model, optimizer, loss, trainSet, devSet, epoches)
    #         # Getting the command.
    #         input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
    #     elif cmd == 'E':
    #         try:
    #             # Loading the model.
    #             model.load_state_dict(torch.load('./Deep_Neural_Network_with_Pytorch/Convolutional_Neural_Network_Demo/Hand_Writting_Number_Classifier/HandWrittingNumberModel.pt'))
    #             # Sending the model into the corresponding device.
    #             model = model.to(device)
    #             # Converting the model mode.
    #             model.eval()
    #             # Testing the model.
    #             # Getting the root of the testing model.
    #             root = './Deep_Neural_Network_with_Pytorch/Convolutional_Neural_Network_Demo/Hand_Writting_Number_Classifier/Test_Image/'
    #             # Getting the testing data.
    #             for filename in os.listdir(root):
    #                 # Getting the image.
    #                 image = Image.open(os.path.join(root, filename))
    #                 # Setting the transformation.
    #                 transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    #                 # Getting the testing data. [1, 28, 28] -> [1, 1, 28, 28]
    #                 inputData = transform(image.convert("1")).unsqueeze(0)
    #                 # Sending the data into the corresponding device.
    #                 inputData = inputData.to(device)
    #                 # Testing the data.
    #                 prediction = model(inputData)
    #                 # Getting the predicted label.
    #                 prediction = int(torch.argmax(prediction, 1))
    #                 # Printing the predicted label.
    #                 print("The predicted label of " + filename + " is: " + str(prediction))
    #                 #Drawing the testing image.
    #                 plt.imshow(image)
    #                 plt.show()
    #                 # Getting the command.
    #                 cmd = input("Input 'Exit' to stop the testing: ")
    #                 # Handling the command.
    #                 if cmd == 'Exit':
    #                     break
    #                 else:
    #                     continue
    #             # Giving the hint.
    #             print("Testing completed!!!")
    #         except:
    #             # Giving the hint.
    #             print("There are not any trained model, please training one first!!!")
    #             cmd = 'T'
    #     else:
    #         cmd = input("Invalid Input! Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")