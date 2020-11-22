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
# The default value of the class size.
Cfg.cs = 10
# The default value of the learning rate.
Cfg.lr = 0.1
# The default value of the batch size.
Cfg.bs = 32
# The default value of the epoches.
Cfg.epoches = 10
# The default value of the random seed.
Cfg.seed = 0
# The default value of the GPU ID.
Cfg.GPUID = -1

# Setting the default values for the directories.
# The default value of the model directory.
Cfg.modelDir = os.path.join('./', 'Checkpoints')
# The default value of the log directory.
Cfg.logDir = os.path.join('./', 'Logs')
# The default value of the dataset directory.
Cfg.dataDir = os.path.join('./', 'Data')

# Setting the arguments' parser.
def argParse():
    # Getting the configurator.
    CFG = Cfg
    # Creating the arguments' parser.
    parser = argparse.ArgumentParser(description = 'Argument Parser')
    # Setting the arguments.
    parser.add_argument('-cs', '--classSize', type = int, dest = 'cs', default = CFG.cs, help = 'Integer => [2, Infinite)')
    parser.add_argument('-lr', '--learningRate', type = float, dest = 'lr', default = CFG.lr, help = 'Float => [0, 1]')
    parser.add_argument('-bs', '--batchSize', type = int, dest = 'bs', default = CFG.bs, help = 'Integer => [1, Infinite)')
    parser.add_argument('-ep', '--epoches', type = int, dest = 'epoches', default = CFG.epoches, help = 'Integer => [1, Infinite)')
    parser.add_argument('-seed', '--seed', type = int, dest = 'seed', default = CFG.seed, help = 'Integer => [0, Infinite)')
    parser.add_argument('-gpu', '--GPUID', type = int, dest = 'GPUID', default = CFG.GPUID, help = 'Integer => [0, Infinite)')
    parser.add_argument('-modelDir', '--modelDir', type = str, dest = 'modelDir', default = CFG.modelDir, help = 'String')
    parser.add_argument('-logDir', '--logDir', type = str, dest = 'logDir', default = CFG.logDir, help = 'String')
    parser.add_argument('-dataDir', '--dataDir', type = str, dest = 'dataDir', default = CFG.dataDir, help = 'String')
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
        Class Size:         {Cfg.cs}
        Learning Rate:      {Cfg.lr}
        Batch Size:         {Cfg.bs}
        Epoches:            {Cfg.epoches}
        Random Seed:        {Cfg.seed}
        GPU ID:             {Cfg.GPUID}
        Model Directory:    {Cfg.modelDir}
        Log Directory:      {Cfg.logDir}
        Dataset Directory:  {Cfg.dataDir}
    ''')
    # Updateing the configurator.
    Cfg = argParse()
    # Printing the configurator's items.
    print(f'''
        Class Size:         {Cfg.cs}
        Learning Rate:      {Cfg.lr}
        Batch Size:         {Cfg.bs}
        Epoches:            {Cfg.epoches}
        Random Seed:        {Cfg.seed}
        GPU ID:             {Cfg.GPUID}
        Model Directory:    {Cfg.modelDir}
        Log Directory:      {Cfg.logDir}
        Dataset Directory:  {Cfg.dataDir}
    ''')