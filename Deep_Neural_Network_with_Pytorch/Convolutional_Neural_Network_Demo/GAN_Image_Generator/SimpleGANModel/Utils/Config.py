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

# Setting the hyperparameters default value.
# The default value of the generator's learning rate.
Cfg.lrG = 0.1
Cfg.lrD = 0.1
# The default value of the latent size.
Cfg.lt = 50
# The default value of the image size.
Cfg.im = 30
Cfg.imageSize = Cfg.im * Cfg.im
# The default value of the epoches.
Cfg.epoches = 10
# The default value of the batch size.
Cfg.bs = 64
# The default value of the random seed.
Cfg.seed = 0
# The default value of the GPU ID.
Cfg.GPUID = -1

# Setting the directories default value.
# The default directory of model.
Cfg.modelDir = os.path.join('./', 'Checkpoints')
# The default directory of logging.
Cfg.logDir = os.path.join('./', 'Logs')
# The default directory of datasets.
Cfg.dataDir = os.path.join('./', 'Data')

# Setting the arguments parser.
def argParse():
    # Getting the configurator.
    CFG = Cfg
    # Setting the arguments parser.
    parser = argparse.ArgumentParser(description = 'Argument Parser')
    # Setting the arguments.
    parser.add_argument('-lrG', '--learningRateG', type = float, dest = 'lrG', default = CFG.lrG, help = 'Float => [0, 1]')
    parser.add_argument('-lrD', '--learningRateD', type = float, dest = 'lrD', default = CFG.lrD, help = 'Float => [0, 1]')
    parser.add_argument('-lt', '--latentSize', type = int, dest = 'lt', default = CFG.lt, help = 'Integer => [1, Infinite)')
    parser.add_argument('-im', '--imageSize', type = int, dest = 'im', default = CFG.im, help = 'Integer => [1, Infinite)')
    parser.add_argument('-ep', '--epoches', type = int, dest = 'epoches', default = CFG.epoches, help = 'Integer => [1, Infinite)')
    parser.add_argument('-bs', '--batchSize', type = int, dest = 'bs', default = CFG.bs, help = 'Integer => [1, Infinite)')
    parser.add_argument('-seed', '--seed', type = int, dest = 'seed', default = CFG.seed, help = 'Integer => [0, Infinite)')
    parser.add_argument('-gpu', '--GPUID', type = int, dest = 'GPUID', default = CFG.seed, help = 'Integer => [0, Infinite)')
    parser.add_argument('-modelDir', '--modelDir', type = str, dest = 'modelDir', default = CFG.modelDir, help = 'String')
    parser.add_argument('-logDir', '--logDir', type = str, dest = 'logDir', default = CFG.logDir, help = 'String')
    parser.add_argument('-dataDir', '--dataDir', type = str, dest = 'dataDir', default = CFG.dataDir, help = 'String')
    # Parsing the arguments.
    args = vars(parser.parse_args())
    args['imageSize'] = args['im'] * args['im']
    # Updating the configurator.
    CFG.update(args)
    # Returning the configurator.
    return Config(CFG)

# Testing the configurator.
if __name__ == "__main__":
    # Printing the items in configuration files.
    print(f'''
        Generator Learning Rate:        {Cfg.lrG}
        Discriminator Learning Rate:    {Cfg.lrD}
        Latent Size:                    {Cfg.lt}
        Image Length:                   {Cfg.im}
        Image Size:                     {Cfg.imageSize}
        Epoches:                        {Cfg.epoches}
        Batch Size:                     {Cfg.bs}
        Random Sedd:                    {Cfg.seed}
        GPU ID:                         {Cfg.GPUID}
        Model Directory:                {Cfg.modelDir}
        Logging Directory:              {Cfg.logDir}
        Dataset Directory:              {Cfg.dataDir}    
    ''')
    # Updating the configurator.
    Cfg = argParse()
    # Printing the items in configuration files.
    print(f'''
        Generator Learning Rate:        {Cfg.lrG}
        Discriminator Learning Rate:    {Cfg.lrD}
        Latent Size:                    {Cfg.lt}
        Image Length:                   {Cfg.im}
        Image Size:                     {Cfg.imageSize}
        Epoches:                        {Cfg.epoches}
        Batch Size:                     {Cfg.bs}
        Random Sedd:                    {Cfg.seed}
        GPU ID:                         {Cfg.GPUID}
        Model Directory:                {Cfg.modelDir}
        Logging Directory:              {Cfg.logDir}
        Dataset Directory:              {Cfg.dataDir}    
    ''')