#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/22
#   File Name:      DataPreprocessor.py
#   Description:    This file is used to setting the data preprocess component.
#============================================================================================#

# Importing the necessary library.
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Creating the class to getting the training and development data.
class dataLoader():
    # Creating the method to get the training and development data.
    @staticmethod
    def Datasets(root, batchSize):
        # Setting the normalization.
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        # Getting the training data.
        trainData = DataLoader(
            datasets.ImageFolder(
                root = root + '/train/',
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
                root = root + '/val/',
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