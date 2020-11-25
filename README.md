# Pytorch_Practice
 Pytorch studying and practicing for Deep Learning

 Datasets Description:

 In this project, the necessary datasets are MNIST, IMDB and CELEBA.

 The MNIST and IMDB could be directly downloaded from the torchvision and torchtext.

 The CELEBA could be downloaded from the Google official: https://drive.google.com/drive/folders/0B7EVK8r0v71pbWNEUjJKdDQ3dGc

 For other datasets, here is the baidu cloud's link: https://pan.baidu.com/s/1-nmxzd4Wm7ijRpUMzRnQQA, code: rbo5

 The above link only has the datasets for Recurrent Neural Network project.

 The remaining datasets of Convolutional Neural Network would not be offered. (Neural Style Transfer and Resnet Transfer Learning)


 Library Description:

 pip install -U setuptools -r requirements.txt
 
 Hint: Some words split tools should be extra downloaded or setted from the internet.


 Project Description:

 All the projects in Deep folders have the Config.py file, which contains all the hyperparameters and directories settings.

 Directly running the Trainer.py file, all the hyperparameters and directories would be applied as the default value.

 There are two ways to changing the settings, one is directly modifying the default value in Config.py file.

 Another one is giving the new value by running the instructions like below:

    python .../Trainer.py -gpu [GPUID] -bs [BatchSize] - lr [LearningRate] -modelDir [Checkpoints Saving Dir] ...
