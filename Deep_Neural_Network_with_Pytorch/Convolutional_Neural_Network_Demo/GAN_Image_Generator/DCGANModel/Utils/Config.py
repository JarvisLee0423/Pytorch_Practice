#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/21
#   File Name:      Config.py
#   Description:    This file is used to set the default value for all the hyperparameters
#                   and necessary configuration.
#============================================================================================#

# Importing the library.
import os
import argparse
from easydict import EasyDict as Config

# Creating the configurator.
Cfg = Config()

# Setting the default value of the hyperparameters.
# The value of the learning rate for generator.
Cfg.lrG = 2e-4
# The value of the learning rate for discriminator.
Cfg.lrD = 2e-4
# The value of the beta one for adam optimization method.
Cfg.beta1 = 0.5
# The value of the bate two for adam optimization method.
Cfg.beta2 = 0.999
# The value of the batch size.
Cfg.bs = 64
# The value of the latent size.
Cfg.lt = 100
# The value of the epoch.
Cfg.epoches = 5
# The value of the GPU ID.
Cfg.GPUID = -1
# The value of the random seed.
Cfg.seed = 1

# Setting the default directories.
# The directory for model saving.
Cfg.modelDir = os.path.join('./', 'Checkpoints')
# The directory for logging file directories.
Cfg.logDir = os.path.join('./', 'Logs')
# The directory for datasets.
Cfg.dataDir = os.path.join('./', 'Data')

# Creating the argument parser.
def argParse():
    # Getting the configurator.
    CFG = Cfg
    # Setting the arguments parser.
    parser = argparse.ArgumentParser(description = 'Argument Parser')
    # Adding the arguments.
    parser.add_argument('-lrG', '--learningRateG', type = float, dest = 'lrG', default = CFG.lrG, help = 'Float => [0, 1]')
    parser.add_argument('-lrD', '--learningRateD', type = float, dest = 'lrD', default = CFG.lrD, help = 'Float => [0, 1]')
    parser.add_argument('-beta1', '--betaOne', type = float, dest = 'beta1', default = CFG.beta1, help = 'Float => [0, 1]')
    parser.add_argument('-beta2', '--betaTwo', type = float, dest = 'beta2', default = CFG.beta2, help = 'Float => [0, 1]')
    parser.add_argument('-bs', '--batchSize', type = int, dest = 'bs', default = CFG.bs, help = 'Integer => [1, Infinite)')
    parser.add_argument('-lt', '--letantSize', type = int, dest = 'lt', default = CFG.lt, help = 'Integer => [1, Infinite)')
    parser.add_argument('-ep', '--epoches', type = int, dest = 'epoches', default = CFG.epoches, help = 'Integer => [1, Infinite)')
    parser.add_argument('-gpu', '--gpu', type = int, dest = 'GPUID', default = CFG.GPUID, help = 'Integer => [0, Infinite)')
    parser.add_argument('-seed', '--seed', type = int, dest = 'seed', default = CFG.seed, help = 'Integer => [0, Infinite)')
    parser.add_argument('-modelDir', '--modelDir', type = str, dest = 'modelDir', default = CFG.modelDir, help = 'String')
    parser.add_argument('-logDir', '--logDir', type = str, dest = 'logDir', default = CFG.logDir, help = 'String')
    parser.add_argument('-dataDir', '--dataDir', type = str, dest = 'dataDir', default = CFG.dataDir, help = 'String')
    # Parsing the arguments.
    args = vars(parser.parse_args())
    # Updating the configurator.
    CFG.update(args)
    # Returning the configurator.
    return Config(CFG)

# Testing the values in the configuration files.
if __name__ == "__main__":
    # Printing the items in configuration files.
    print(f'''
    Generator Learning Rate:        {Cfg.lrG}
    Discriminator Learning Rate:    {Cfg.lrD}
    Adam Beta One:                  {Cfg.beta1}
    Adam Beta Two:                  {Cfg.beta2}
    Batch Size:                     {Cfg.bs}
    Latent Size:                    {Cfg.lt}
    Epoches:                        {Cfg.epoches}
    GPU ID:                         {Cfg.GPUID}
    Random Seed:                    {Cfg.seed}
    Model Directory:                {Cfg.modelDir}
    Log Directory:                  {Cfg.logDir}
    Dataset Directory:              {Cfg.dataDir}
    ''')
    # Getting the configurator.
    cfg = argParse()
    # Printing the items in configuration files.
    print(f'''
    Generator Learning Rate:        {cfg.lrG}
    Discriminator Learning Rate:    {cfg.lrD}
    Adam Beta One:                  {cfg.beta1}
    Adam Beta Two:                  {cfg.beta2}
    Batch Size:                     {cfg.bs}
    Latent Size:                    {cfg.lt}
    Epoches:                        {cfg.epoches}
    GPU ID:                         {Cfg.GPUID}
    Random Seed:                    {cfg.seed}
    Model Directory:                {cfg.modelDir}
    Log Directory:                  {cfg.logDir}
    Dataset Directory:              {cfg.dataDir}
    ''')