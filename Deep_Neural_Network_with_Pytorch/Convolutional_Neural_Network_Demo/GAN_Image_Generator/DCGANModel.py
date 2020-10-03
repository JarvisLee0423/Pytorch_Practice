#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/09/21
#   Project Name:       DCGANModel.py
#   Description:        Build a deconvolutional GAN model to generating the image in 
#                       MNIST datasets.
#   Model Description:  Input               ->  MNIST Images and Fake Images
#                       Generator           ->  ConvTranspose2d     ->  InChannel:  100
#                                                                   ->  OutChannel: 1024
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     1
#                                                                   ->  Padding:    0
#                                           ->  BatchNorm2d         ->  InChannel:  1024
#                                           ->  ReLu                ->  InPlace:    True
#                                           ->  ConvTranspose2d     ->  InChannel:  1024
#                                                                   ->  OutChannel: 512
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     2
#                                                                   ->  Padding:    1
#                                           ->  BatchNorm2d         ->  InChannel:  512
#                                           ->  ReLu                ->  InPlace:    True
#                                           ->  ConvTranspose2d     ->  InChannel:  512
#                                                                   ->  OutChannel: 256
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     2
#                                                                   ->  Padding:    1
#                                           ->  BatchNorm2d         ->  InChannel:  256
#                                           ->  ReLu                ->  InPlace:    True                                    
#                                           ->  ConvTranspose2d     ->  InChannel:  256
#                                                                   ->  OutChannel: 128
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     2
#                                                                   ->  Padding:    1
#                                           ->  BatchNorm2d         ->  InChannel:  128
#                                           ->  ReLu                ->  InPlace:    True
#                                           ->  ConvTranspose2d     ->  InChannel:  128
#                                                                   ->  OutChannel: 3
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     2
#                                                                   ->  Padding:    1
#                                           ->  Tanh
#                       Discriminator       ->  Conv2d              ->  InChannel:  3
#                                                                   ->  OutChannel: 128
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     2
#                                                                   ->  Padding:    1
#                                           ->  BatchNorm2d         ->  InChannel:  128
#                                           ->  LeakyReLu           ->  Rate:       0.2
#                                                                   ->  InPlace:    True
#                                           ->  Conv2d              ->  InChannel:  128
#                                                                   ->  OutChannel: 256
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     2
#                                                                   ->  Padding:    1
#                                           ->  BatchNorm2d         ->  InChannel:  256
#                                           ->  LeakyReLu           ->  Rate:       0.2
#                                                                   ->  InPlace:    True
#                                           ->  Conv2d              ->  InChannel:  256
#                                                                   ->  OutChannel: 512
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     2
#                                                                   ->  Padding:    1
#                                           ->  BatchNorm2d         ->  InChannel:  512
#                                           ->  LeakyReLu           ->  Rate:       0.2
#                                                                   ->  InPlace:    True
#                                           ->  Conv2d              ->  InChannel:  512
#                                                                   ->  OutChannel: 1024
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     2
#                                                                   ->  Padding:    1
#                                           ->  BatchNorm2d         ->  InChannel:  1024
#                                           ->  LeakyReLu           ->  Rate:       0.2
#                                                                   ->  InPlace:    True
#                                           ->  Conv2d              ->  InChannel:  1024
#                                                                   ->  OutChannel: 1
#                                                                   ->  Kernel:     (4, 4)
#                                                                   ->  Stride:     1
#                                                                   ->  Padding:    0
#                                           ->  Sigmoid
#============================================================================================#

# Importing the necessary library.
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Fixing the device and random seed.
if torch.cuda.is_available():
    # Fixing the device.
    torch.cuda.set_device(0)
    # Fixing the random seed.
    #torch.cuda.manual_seed(1)
    # Setting the device.
    device = 'cuda'
else:
    # Fixing the random seed.
    #torch.manual_seed(1)
    # Setting the device.
    device = 'cpu'

# Setting the hyperparameters.
# The value of the learning rate.
learningRateG = 2e-4
learningRateD = 2e-4
# The value of the batch size.
batchSize = 64
# The value of the latent size.
latentSize = 100
# The value of the epoch.
epoches = 5

# Creating the dataloader.
class dataLoader():
    # Creating the method to get the train data.
    @staticmethod
    def CELEBA(batchSize):
        # Setting the data root.
        root = './Datasets/CELEBA/'
        # Setting the transformation method.
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = (0.5, 0.5, 0.5),
                std = (0.5, 0.5, 0.5)
            )
        ])
        # Getting the data.
        trainData = datasets.ImageFolder(
            root = root,
            transform = transform
        )
        # Getting the training set.
        trainSet = DataLoader(
            trainData,
            batch_size = batchSize,
            shuffle = True,
            drop_last = True
        )
        # Returning the train set.
        return trainSet

# Creating the model for generator.
class Generator(nn.Module):
    # Creating the constructor.
    def __init__(self, latentSize):
        # Inheritting the super constructor.
        super(Generator, self).__init__()
        # Setting the model.
        self.convTrans2d1 = nn.ConvTranspose2d(latentSize, 1024, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(1024)
        self.convTrans2d2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.convTrans2d3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.convTrans2d4 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.convTrans2d5 = nn.ConvTranspose2d(128, 3, 4, 2, 1)
    # Setting the method to initializing the weight.
    @staticmethod
    def weightInit(model):
        # Getting the model name.
        name = model.__class__.__name__
        # Initializing the weight.
        if name.find("Conv") != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
            nn.init.constant_(model.bias.data, 0)
        elif name.find("BatchNorm") != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
            nn.init.constant_(model.bias.data, 0)
    # Doing the forward propagation.
    def forward(self, x):
        # [batchSize, 100, 1, 1] -> [batchSize, 1024, 4, 4]
        x = self.convTrans2d1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace = True)
        # [batchSize, 1024, 4, 4] -> [batchSize, 512, 8, 8]
        x = self.convTrans2d2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace = True)
        # [batchSize, 512, 8, 8] -> [batchSize, 256, 16, 16]
        x = self.convTrans2d3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace = True)
        # [batchSize, 256, 16, 16] -> [batchSize, 128, 32, 32]
        x = self.convTrans2d4(x)
        x = self.bn4(x)
        x = F.relu(x, inplace = True)
        # [batchSize, 128, 32, 32] -> [batchSize, 3, 64, 64]
        x = self.convTrans2d5(x)
        # Returning the data.
        return torch.tanh(x)

# Creating the model for discrimitor.
class Discrimitor(nn.Module):
    # Creating the constructor.
    def __init__(self):
        # Inheritting the super constructor.
        super(Discrimitor, self).__init__()
        # Setting the model.
        self.conv2d1 = nn.Conv2d(3, 128, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2d2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv2d3 = nn.Conv2d(256, 512, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv2d4 = nn.Conv2d(512, 1024, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.conv2d5 = nn.Conv2d(1024, 1, 4, 1, 0)
    # Doing the forward propagation.
    def forward(self, x):
        # [batchSize, 3, 64, 64] -> [batchSize, 128, 32, 32]
        x = self.conv2d1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2, inplace = True)
        # [batchSize, 128, 32, 32] -> [batchSize, 256, 16, 16]
        x = self.conv2d2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2, inplace = True)
        # [batchSize, 256, 16, 16] -> [batchSize, 512, 8, 8]
        x = self.conv2d3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2, inplace = True)
        # [batchSize, 512, 8, 8] -> [batchSize, 1024, 4, 4]
        x = self.conv2d4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2, inplace = True)
        # [batchSize, 1024, 4, 4] -> [batchSize, 1, 1, 1]
        x = self.conv2d5(x)
        # [batchSize, 1, 1, 1] -> [batchSize, 1]
        x = x.squeeze().unsqueeze(1)
        # Returning the data.
        return torch.sigmoid(x)

# Creating the model for DCGAN.
class DCGANModelNN():
    # Creating the method for training.
    @staticmethod
    def trainer(trainData, latentSize, batchSize, learningRateG, learningRateD, epoches):
        # Creating the model.
        G = Generator(latentSize)
        D = Discrimitor()
        # Sending the models into correspoding device.
        G = G.to(device)
        D = D.to(device)
        # Reinitializing the generator's weight and bias.
        G.apply(Generator.weightInit)
        # Setting the loss function.
        loss = nn.BCELoss()
        # Setting the optimizers.
        optimG = optim.Adam(G.parameters(), lr = learningRateG, betas = (0.5, 0.999))
        optimD = optim.Adam(D.parameters(), lr = learningRateD, betas = (0.5, 0.999))
        # Setting the truth label and fake label.
        trueLabel = torch.ones(batchSize, 1).to(device)
        fakeLabel = torch.zeros(batchSize, 1).to(device)
        # Training the model.
        for epoch in range(epoches):
            # Getting each iterations' loss.
            trainLossG = []
            trainLossD = []
            for i, (data, _) in  enumerate(trainData):
                # Sending the data into the corresponding device.
                data = data.to(device)
                # Creating the lantent space.
                latentSpace = torch.randn(batchSize, latentSize, 1, 1).to(device)
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

                # Creating the lantent space.
                latentSpace = torch.randn(batchSize, latentSize, 1, 1).to(device)
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

                # Printing the training accuracy.
                if i % 200 == 0:
                    print("The training accuracy of Epoch [%d/%d] Iteration: [%d/%d]: G Loss [%.4f] || D Loss [%.4f] || G Accuracy [%.4f] || D Accuracy True[%.4f] Fake[%.4f]" % (epoch + 1, epoches, i + 1, len(trainData), (np.sum(trainLossG) / len(trainLossG)), (np.sum(trainLossD) / len(trainLossD)), generatedPrediction.mean().item(), truePrediction.mean().item(), fakePrediction.mean().item()))
                    # Getting each iterations' loss.
                    trainLossG = []
                    trainLossD = []
            # Saving the model.
            torch.save(G.train().state_dict(), './Deep_Neural_Network_with_Pytorch/Convolutional_Neural_Network_Demo/GAN_Image_Generator/DCGANGenerator.pt')

if __name__ == "__main__":
    #Getting the data.
    trainData = dataLoader.CELEBA(batchSize)
    # Outputing the data.
    for _, (images, _) in enumerate(trainData):
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
            DCGANModelNN.trainer(trainData, latentSize, batchSize, learningRateG, learningRateD, epoches)
            # Getting the command.
            cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
        elif cmd == 'E':
            try:
                # Loading the model.
                model = Generator(latentSize)
                model.load_state_dict(torch.load('./Deep_Neural_Network_with_Pytorch/Convolutional_Neural_Network_Demo/GAN_Image_Generator/DCGANGenerator.pt'))
                # Sending the model into the corresponding device.
                model = model.to(device).eval()
                # Creating the latent space.
                latentSpace = torch.randn(batchSize, latentSize, 1, 1).to(device)
                # Getting the fake image.
                fakeImages = model(latentSpace).to('cpu')
                # Plotting the image.
                while True:
                    # Getting the image number.
                    i = input("Please input a image number (%d <= number <= %d and 'Exit' for quit): " % (1, batchSize))
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