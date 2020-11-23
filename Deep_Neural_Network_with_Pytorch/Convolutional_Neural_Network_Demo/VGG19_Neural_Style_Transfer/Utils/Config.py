#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/22
#   File Name:      Config.py
#   Description:    This file is used to setting the hyperparameters and directories.
#============================================================================================#

# Importing the necessary library.
import os
import argparse
from easydict import EasyDict as Config

# Creating the configurator.
Cfg = Config()

# Setting the default values for the hyperparameters.
# The default value of the learning rate.
Cfg.lr = 3e-3
# The default value of the momentum.
Cfg.momentum = 0.9
# The default value of the weight decay.
Cfg.wd = 5e-5
# The default value of the content weight.
Cfg.cw = 150
# The default value of the style weight.
Cfg.sw = 1000
# The default value of the epoches.
Cfg.epoches = 5000
# The default value of the random seed.
Cfg.seed = 1
# The default value of the GPU ID.
Cfg.GPUID = -1

# Setting the default values for the directories.
# The default value of the log directory.
Cfg.logDir = os.path.join('./', 'Logs')
# The default value of the dataset directory.
Cfg.dataDir = os.path.join('./', 'Data')
# The default name for the style image.
Cfg.styleIM = 'style.jpg'
# The default name for the content image.
Cfg.contentIM = 'content.jpg'

# Setting the arguments' parser.
def argParse():
    # Getting the configurator.
    CFG = Cfg
    # Creating the arguments' parser.
    parser = argparse.ArgumentParser(description = 'Argument Parser')
    # Setting the arguments.
    parser.add_argument('-lr', '--learningRate', type = float, dest = 'lr', default = CFG.lr, help = 'Float => [0, 1]')
    parser.add_argument('-momentum', '--momentum', type = float, dest = 'momentum', default = CFG.momentum, help = 'Float => [0, 1]')
    parser.add_argument('-wd', '--weightDecay', type = float, dest = 'wd', default = CFG.wd, help = 'Float => [0, 1]')
    parser.add_argument('-cw', '--contentWeight', type = int, dest = 'cw', default = CFG.cw, help = 'Integer => [1, Infinite)')
    parser.add_argument('-sw', '--styleWeight', type = int, dest = 'sw', default = CFG.sw, help = 'Integer => [1, Infinite)')
    parser.add_argument('-ep', '--epoches', type = int, dest = 'epoches', default = CFG.epoches, help = 'Integer => [1, Infinite)')
    parser.add_argument('-seed', '--seed', type = int, dest = 'seed', default = CFG.seed, help = 'Integer => [0, Infinite)')
    parser.add_argument('-gpu', '--GPUID', type = int, dest = 'GPUID', default = CFG.GPUID, help = 'Integer => [0, Infinite)')
    parser.add_argument('-logDir', '--logDir', type = str, dest = 'logDir', default = CFG.logDir, help = 'String')
    parser.add_argument('-dataDir', '--dataDir', type = str, dest = 'dataDir', default = CFG.dataDir, help = 'String')
    parser.add_argument('-contentIM', '--contentIM', type = str, dest = 'contentIM', default = CFG.contentIM, help = 'String')
    parser.add_argument('-styleIM', '--styleIM', type = str, dest = 'styleIM', default = CFG.styleIM, help = 'String')
    # Parsing the argument.
    args = vars(parser.parse_args())
    # Updating the configurator.
    CFG.update(args)
    # Returning the configurator.
    return CFG

# Testing the configurator.
if __name__ == "__main__":
    # Printing the configurator's items.
    print(f'''
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
    # Updateing the configurator.
    Cfg = argParse()
    # Printing the configurator's items.
    print(f'''
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