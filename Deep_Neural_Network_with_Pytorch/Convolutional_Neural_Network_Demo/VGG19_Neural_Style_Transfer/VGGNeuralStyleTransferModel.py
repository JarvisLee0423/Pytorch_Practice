#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/08/27
#   Project Name:       VGGNeuralStyleTransferModel.py
#   Description:        Apply the VGG-19 net to complete the neural style transfer.
#   Model Description:  Input               ->  Content, Style and Target Image
#                       VGG-19              ->  Training the target image make its content
#                                               looks like content image and style looks like
#                                               style image.
#============================================================================================#

# Importing the necessary library.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

# Fixing the computer device and random seed.
if torch.cuda.is_available():
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
# The value of the learning rate.
learningRate = 0.003
# The value of the momentum.
momentum = 0.9
# The value of the weight decay.
weightDecay = 0.00005
# The value of the content weight.
contentWeight = 150
# The value of the style weight.
styleWeight = 1000
# The value of the epoch.
epoches = 5000

# Creating the image processor.
class dataProcessor():
    # Creating the method to preprocess the image.
    @staticmethod
    def imageProcessor(image):
        # Setting the transform method.
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406], 
                std = [0.229, 0.224, 0.225]
            )
        ])
        # Getting the image tensor.
        image = transform(image)
        # Returning the image.
        return image.unsqueeze(0)

# Creating the model.
class VGGNeuralStyleTransferModelNN(nn.Module):
    # Creating the constructor.
    def __init__(self):
        # Inheriting the super constructor.
        super(VGGNeuralStyleTransferModelNN, self).__init__()
        # Getting the VGG-19 as the feature extractor.
        self.vgg19 = models.vgg19(pretrained = True).features
        # Getting the indices of the features which would be used.
        self.features = ['0', '5', '10', '19', '28']
    # Defining the forward.
    def forward(self, image):
        # Initializing the feature list.
        feature = []
        # Getting the each layers of the vgg19.
        for name, layer in self.vgg19._modules.items():
            # Feeding the data.
            image = layer(image)
            # Getting the required features.
            if name in self.features:
                feature.append(image)
        # Returning the features.
        return feature
    # Defining the training method.
    @staticmethod
    def trainer(content, style, epoches = 2000, learningRate = 0.003, momentum = 0.9, weightDecay = 0.00005, contentWeight = 100, styleWeight = 2000):
        # Initializing the target image.
        target = torch.rand_like(content).to(device).requires_grad_(True)
        #target = content.clone().requires_grad_(True)
        # Creating the model.
        model = VGGNeuralStyleTransferModelNN().to(device).eval()
        # Creating the optimizer.
        optimizer = optim.SGD([target], lr = learningRate, momentum = momentum, weight_decay = weightDecay)
        # Training the model.
        for epoch in range(epoches):
            # Initializing the content and style loss.
            contentLoss = 0
            styleLoss = 0
            # Getting the features of the content, style and target image.
            ft = model(target)
            fc = model(content)
            fs = model(style)
            # Computing the loss.
            for f1, f2, f3 in zip(ft, fc, fs):
                # Computing the content loss.
                contentLoss += torch.mean((f1 - f2) ** 2)
                # Resize the feature map. [b, c, h, w] -> [c, h * w]
                b, c, h, w = f1.shape
                f1 = f1.resize(c, h * w)
                f2 = f2.resize(c, h * w)
                f3 = f3.resize(c, h * w)
                # Getting the gram matrix. [c, h * w] -> [c, c]
                f1 = torch.mm(f1, f1.t())
                f3 = torch.mm(f3, f3.t())
                # Computing the style loss.
                styleLoss += torch.mean((f1 - f3) ** 2) / (c * h * w)
            # Getting the final loss.
            loss = contentWeight * contentLoss + styleWeight * styleLoss
            # Clearing the previous gradient.
            optimizer.zero_grad()
            # Computing the backward.
            loss.backward()
            # Updating the target.
            optimizer.step()
            # Getting the training information.
            if epoch % 100 == 0:
                print('The epoch ' + str(epoch + 1) + '/' + str(epoches) + ' training: Loss = ' + str(loss.item()))
                print('The epoch ' + str(epoch + 1) + '/' + str(epoches) + ' training: Content Loss = ' + str(contentLoss.item()) + ' || Style Loss = ' + str(styleLoss.item()))
        # Return the target image.
        return target.to('cpu')

# Training the model.
if __name__ == "__main__":
    pass
    # # Getting the images.
    # content = Image.open('./Deep_Neural_Network_with_Pytorch/Convolutional_Neural_Network_Demo/VGG19_Neural_Style_Transfer/Image/Content.jpg')
    # style = Image.open('./Deep_Neural_Network_with_Pytorch/Convolutional_Neural_Network_Demo/VGG19_Neural_Style_Transfer/Image/Style.jpg')
    # plt.imshow(content)
    # plt.show()
    # plt.imshow(style)
    # plt.show()
    # # Preprocessing the images.
    # content = dataProcessor.imageProcessor(content)
    # style = dataProcessor.imageProcessor(style)
    # # Setting the transform.
    # transform = transforms.ToPILImage()
    # # Plotting the images.
    # show = transform(content.squeeze())
    # plt.imshow(show)
    # plt.show()
    # show = transform(style.squeeze())
    # plt.imshow(show)
    # plt.show()
    # # Training the model.
    # target = VGGNeuralStyleTransferModelNN.trainer(content.to(device), style.to(device), epoches, learningRate, momentum, weightDecay, contentWeight, styleWeight)
    # # Getting the denormalization transformation.
    # denormalization = transforms.Normalize([-2.12, -2.04, -1.80], [4.37, 4.46, 4.44])
    # # Denormalizing the target image.
    # target = denormalization(target.squeeze()).clamp_(0, 1)
    # # Plotting the target image.
    # target = transform(target)
    # plt.figure('Target Image')
    # plt.imshow(target)
    # plt.savefig('./Deep_Neural_Network_with_Pytorch/Convolutional_Neural_Network_Demo/VGG19_Neural_Style_Transfer/Image/Target.jpg')
    # plt.show()