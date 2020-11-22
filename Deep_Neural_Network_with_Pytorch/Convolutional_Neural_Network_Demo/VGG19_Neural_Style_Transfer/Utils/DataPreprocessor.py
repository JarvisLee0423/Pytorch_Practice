#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/22
#   File Name:      DataPreprocessor.py
#   Description:    This file is used to setting the data preprocess component.
#============================================================================================#

# Importing the necessary library.
from torchvision import transforms, datasets

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