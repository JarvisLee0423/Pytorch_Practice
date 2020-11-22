#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/11/22
#   Project Name:       VGGNeuralStyleTransferModel.py
#   Description:        Apply the VGG-19 net to complete the neural style transfer.
#   Model Description:  Input               ->  Content, Style and Target Image
#                       VGG-19              ->  Training the target image make its content
#                                               looks like content image and style looks like
#                                               style image.
#============================================================================================#

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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