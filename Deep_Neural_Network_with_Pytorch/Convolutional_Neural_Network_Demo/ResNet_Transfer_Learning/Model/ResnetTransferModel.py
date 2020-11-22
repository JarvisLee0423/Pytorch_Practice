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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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