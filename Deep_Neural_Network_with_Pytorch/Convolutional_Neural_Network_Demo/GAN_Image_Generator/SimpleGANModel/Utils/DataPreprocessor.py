#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/22
#   File Name:      DataPreprocessor.py
#   Description:    This file is used to setting the data preprocessor components.
#============================================================================================#

# Importing the necessary library.
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Creating the data loader.
class dataLoader():
    # Defining the MNIST data loader.
    @staticmethod
    def MNIST(root, batchSize):
        # Checking whether download the data.
        if os.path.exists(root + '/MNIST/'):
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
        ])
        # Getting the training data.
        trainData = datasets.MNIST(
            root = root,
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