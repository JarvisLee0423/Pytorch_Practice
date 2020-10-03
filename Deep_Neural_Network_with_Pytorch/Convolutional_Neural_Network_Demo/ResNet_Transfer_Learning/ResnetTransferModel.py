#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/08/26
#   Project Name:       ResnetTransferModel.py
#   Description:        Apply the Resnet 18 to do the transfer learning to process the three
#                       classes classification problem.
#   Model Description:  Input               ->  Image
#                       Resnet              ->  Fine Tunning only the full-connected layer
#                                               would be updating.
#============================================================================================#

# Importing the necessary library.
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image

# Fixing the computer device and random seed.
if torch.cuda.is_available():
    # Fixing the random seed.
    torch.cuda.manual_seed(1)
    # Fixing the computer device.
    torch.cuda.set_device(0)
    # Setting the device.
    device = 'cuda'
else:
    # Fixing the random seed.
    torch.manual_seed(1)
    # Setting the device.
    device = 'cpu'

# Setting the hyperparameters.
# The value of the inChannel.
inChannel = 3
# The value of the class size.
classSize = 3
# The value of the learning rate.
learningRate = 0.01
# The value of the batch size.
batchSize = 64
# The value of the epoches.
epoches = 25

# Creating the class to getting the training and development data.
class dataLoader():
    # Creating the method to get the training and development data.
    @staticmethod
    def NGZK(batchSize):
        # Setting the normalization.
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        # Getting the training data.
        trainData = DataLoader(
            datasets.ImageFolder(
                root = './Datasets/NGZK/train/',
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize
                ])
            ),
            batch_size = batchSize,
            shuffle = True
        )
        # Getting the development data.
        devData = DataLoader(
            datasets.ImageFolder(
                root = './Datasets/NGZK/val/',
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize
                ])
            ),
            batch_size = batchSize,
            shuffle = False
        )
        # Returning the data.
        return trainData, devData

# Creating the model.
class ResnetTransferModelNN():
    # Defining the method to getting the model.
    @staticmethod
    def ResnetInitialization(classSize, preTrained = True, featureExtractor = True):
        # Getting the model.
        model = models.resnet18(pretrained = preTrained)
        # Fixing the weight if the model is used to be the feature extractor.
        if featureExtractor:
            # Fixing the weight.
            for param in model.parameters():
                param.requires_grad = False
        # Getting the number of input features of the fc of the resnet18.
        inFeature = model.fc.in_features
        # Changing the fc layer.
        model.fc = nn.Linear(inFeature, classSize)
        # Returning the model.
        return model
    # Defining the method to training the data.
    @staticmethod
    def trainer(trainSet, devSet, epoches, classSize, learningRate):
        # Creating the model.
        model = ResnetTransferModelNN.ResnetInitialization(classSize)
        # Sending the model into the corresponding device.
        model = model.to(device)
        # Creating the optimizer.
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learningRate, betas = [0.9, 0.999])
        # Creating the learning rate scheduler.
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = len(trainSet), T_mult = 1, eta_min = 0)
        # Creating the loss function.
        loss = nn.CrossEntropyLoss()
        # Initializing the evaluating accuracy.
        evalAccs = []
        # Training the model.
        for epoch in range(epoches):
            # Initializing the train accuracy.
            trainAcc = []
            # Initializing the train loss.
            trainLoss = []
            # Getting the training data.
            for i, (data, label) in enumerate(trainSet):
                # Preparing the data.
                data = Variable(data).to(device)
                label = Variable(label).to(device)
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
            # Evaluating the model.
            evalLoss, evalAcc = ResnetTransferModelNN.evaluator(model.eval(), optimizer, loss, devSet)
            # Selecting the best model.
            if len(evalAccs) == 0 or evalAcc >= max(evalAccs):
                # Saving the model.
                torch.save(model.train().state_dict(), './Deep_Neural_Network_with_Pytorch/Convolutional_Neural_Network_Demo/ResNet_Transfer_Learning/ResnetTransferModel.pt')
                print("Model Saved")
            # Applying the learning rate scheduler.
            else:
                scheduler.step()
                print("Learning Rate Decay -> " + str(scheduler.get_last_lr()))
            # Converting the model mode.
            model.train()
            # Storing the evaluating accuracy.
            evalAccs.append(evalAcc)
            # Printing the training information.
            print("The epoch " + str(epoch + 1) + "/" + str(epoches) + " training: Loss = " + str(np.sum(trainLoss) / len(trainLoss)) + " || Acc = " + str(np.sum(trainAcc) / len(trainAcc)))
            print("The epoch " + str(epoch + 1) + "/" + str(epoches) + " evaluating: Loss = " + str(evalLoss) + " || Acc = " + str(evalAcc))
    # Defining the method to evaluating the model.
    @staticmethod
    def evaluator(model, optimizer, loss, devSet):
        # Initializing the evaluating loss.
        evalLoss = []
        # Initializing the evaluating accuracy.
        evalAcc = []
        # Evaluating the model.
        for i, (data, label) in enumerate(devSet):
            # Preparing the data.
            data = Variable(data).to(device)
            label = Variable(label).to(device)
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

# Training the model.
if __name__ == "__main__":
    # Getting the training data.
    trainSet, devSet = dataLoader.NGZK(batchSize)
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
            ResnetTransferModelNN.trainer(trainSet, devSet, epoches, classSize, learningRate)
            cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
        elif cmd == 'E':
            try:
                # Getting the output label.
                labels = ['堀未央奈', '齋藤飛鳥', '筒井あやめ']
                # Loading the model.
                model = ResnetTransferModelNN.ResnetInitialization(classSize)
                model.load_state_dict(torch.load('./Deep_Neural_Network_with_Pytorch/Convolutional_Neural_Network_Demo/ResNet_Transfer_Learning/ResnetTransferModel.pt'))
                # Sending the model into corresponding device.
                model = model.to(device)
                # Converting the model mode.
                model.eval()
                # Getting the data.
                for _, (data, label) in enumerate(devSet):
                    # Getting the data.
                    for j in range(len(data)):
                        # Preparing the testing data.
                        testData = Variable(data[j]).to(device)
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