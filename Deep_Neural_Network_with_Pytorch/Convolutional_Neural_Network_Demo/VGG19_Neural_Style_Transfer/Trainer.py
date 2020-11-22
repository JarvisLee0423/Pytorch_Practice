#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/22
#   File Name:      Trainer.py
#   Description:    This file is used to training the model.
#============================================================================================#

# Imprting the necessary library.
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from visdom import Visdom
from Model.VGGNeuralStyleTransferModel import VGGNeuralStyleTransferModelNN
from Utils.Config import argParse
from Utils.DataPreprocessor import dataProcessor

# Creating the configurator.
Cfg = argParse()

# Setting the device and random seed.
if torch.cuda.is_available():
    # Setting the device.
    device = 'cuda'
    # Fixing the device.
    if Cfg.GPUID != -1:
        torch.cuda.set_device(Cfg.GPUID)
    # Fixing the random seed.
    np.random.seed(Cfg.seed)
    torch.cuda.manual_seed(Cfg.seed)
else:
    # Setting the device.
    device = 'cpu'
    # Fixing the device.
    np.random.seed(Cfg.seed)
    torch.manual_seed(Cfg.seed)

# Defining the training method.
def trainer(content, style):
    # Creating the logging.
    logging.basicConfig(filename = Cfg.logDir + '/logging.txt', filemode = 'a', level = logging.INFO, format = '%(asctime)s %(levelname)s %(message)s', datefmt = '%Y-%D %H:%M:%S %p')
    # Logging the information.
    logging.info(f'''
        Learning Rate:      {Cfg.lr}
        Momentum:           {Cfg.momentum}
        Weight Decay:       {Cfg.wd}
        Content Weight:     {Cfg.cw}
        Style Weight:       {Cfg.sw}
        Epoches:            {Cfg.epoches}
        Random Seed:        {Cfg.seed}
        GPU ID:             {Cfg.GPUID}
        Log Directory:      {Cfg.logDir}
        Dataset Directory:  {Cfg.dataDir}
        Content Image:      {Cfg.contentIM}
        Style Image:        {Cfg.styleIM}
    ''')
    # Creating the visdom.
    vis = Visdom(env = 'NeuralStyleModel')
    # Creating the graph.
    lossGraph = vis.line(X = [0], Y = [0], opts = dict(legend = ['ContentLoss', 'StyleLoss'], xlabel = 'Epoches', ylabel = 'Loss', title = f'{Cfg.contentIM} and {Cfg.styleIM} Neural Transfer Loss'), name = 'ContentLoss')
    vis.line(X = [0], Y = [0], win = lossGraph, update = 'append', name = 'StyleLoss')
    # Initializing the target image.
    target = torch.rand_like(content).to(device).requires_grad_(True)
    #target = content.clone().requires_grad_(True)
    # Creating the model.
    model = VGGNeuralStyleTransferModelNN().to(device).eval()
    # Creating the optimizer.
    optimizer = optim.SGD([target], lr = Cfg.lr, momentum = Cfg.momentum, weight_decay = Cfg.wd)
    # Training the model.
    for epoch in range(Cfg.epoches):
        # Initializing the content and style loss.
        contentLoss = 0
        styleLoss = 0
        # Getting the features of the content, style and target image.
        ft = model(target)
        fc = model(content)
        fs = model(style)
        # Setting the loading bar.
        with tqdm(total = len(list(zip(ft, fc, fs))), desc = f'Epoch {epoch + 1}/{Cfg.epoches}', unit = 'features', ncols = 100) as pbars:
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

                # Updating the laoding bar.
                pbars.update(1)
        # Closing the loading bar.
        pbars.close()
        # Getting the final loss.
        loss = Cfg.cw * contentLoss + Cfg.sw * styleLoss
        # Clearing the previous gradient.
        optimizer.zero_grad()
        # Computing the backward.
        loss.backward()
        # Updating the target.
        optimizer.step()
        # Drawing the graph.
        vis.line(
            X = [epoch + 1],
            Y = [contentLoss.item()],
            win = lossGraph,
            update = 'append',
            name = 'ContentLoss'
        )
        vis.line(
            X = [epoch + 1],
            Y = [styleLoss.item()],
            win = lossGraph,
            update = 'append',
            name = 'StyleLoss'
        )
        # Logging the information
        logging.info('Epoch [%d/%d] -> Training: Content Loss [%.4f] || Style Loss [%.4f]' % (epoch + 1, Cfg.epoches, contentLoss.item(), styleLoss.item()))
    # Return the target image.
    return target.to('cpu')

# Training the model.
if __name__ == "__main__":
    # Getting the images.
    content = Image.open(os.path.join(Cfg.dataDir, Cfg.contentIM))
    style = Image.open(os.path.join(Cfg.dataDir, Cfg.styleIM))
    plt.imshow(content)
    plt.show()
    plt.imshow(style)
    plt.show()
    # Preprocessing the images.
    content = dataProcessor.imageProcessor(content)
    style = dataProcessor.imageProcessor(style)
    # Setting the transform.
    transform = transforms.ToPILImage()
    # Plotting the images.
    show = transform(content.squeeze())
    plt.imshow(show)
    plt.show()
    show = transform(style.squeeze())
    plt.imshow(show)
    plt.show()
    # Training the model.
    target = trainer(content.to(device), style.to(device))
    # Getting the denormalization transformation.
    denormalization = transforms.Normalize([-2.12, -2.04, -1.80], [4.37, 4.46, 4.44])
    # Denormalizing the target image.
    target = denormalization(target.squeeze()).clamp_(0, 1)
    # Plotting the target image.
    target = transform(target)
    plt.figure('Target Image')
    plt.imshow(target)
    plt.savefig(os.path.join(Cfg.dataDir, 'TargetOf' + Cfg.contentIM.split('.')[0] + '.jpg'))
    plt.show()