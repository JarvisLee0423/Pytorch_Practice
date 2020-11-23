#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/22
#   File Name:      DataPreprocessor.py
#   Description:    This file is used to set the data preprocessor.
#============================================================================================#

# Importing the necessary library.
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Creating the dataloader.
class dataLoader():
    # Creating the method to get the train data.
    @staticmethod
    def CELEBA(root, batchSize):
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