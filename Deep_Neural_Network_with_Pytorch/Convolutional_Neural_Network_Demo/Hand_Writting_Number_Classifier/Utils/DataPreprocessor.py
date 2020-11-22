#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/22
#   File Name:      DatapRreprocessor.py
#   Description:    This file is used to setting the data preprocessor components.
#============================================================================================#

# Importing the necessary library.
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Creating the dataloader.
# Creating the data generator.
class dataLoader():
    # Creating the data generator.
    @staticmethod
    def MNIST(root, batchSize):
        # Indicating whether there are the local dataset or not.
        if os.path.exists(root + '/MNIST/'):
            downloadDataset = False
        else:
            downloadDataset = True
        # Setting the training and development data.
        trainData = datasets.MNIST(
            # Getting the dataset from the root.
            root = root,
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
            root = root,
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