#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/24
#   File Name:      Config.py
#   Description:    This file is used to setting the default value of the hyperparameters
#                   and directories.
#============================================================================================#

# Importing the necessary library.
import os
import argparse
from easydict import EasyDict as Config

# Creating the configurator.
Cfg = Config()

# Setting the default value of the hyperparameters.
# Setting the default value of the vocabulary size.
Cfg.vs = 50000
# Setting the default value of the embedding size.
Cfg.es = 300
# Setting the default value of the class size.
Cfg.cs = 1
# Setting the default value of the learning rate.
Cfg.lr = 0.1
# Setting the default value of the beta1.
Cfg.beta1 = 0.5
# Setting the default value of the beta2.
Cfg.beta2 = 0.9
# Setting the default value of the weight decay.
Cfg.wd = 2e-5
# Setting the default value of the batch size.
Cfg.bs = 128
# Setting the default value of the epoches.
Cfg.epoches = 50
# Setting the default value of the random seed.
Cfg.seed = 0
# Setting the default value of the GPU ID.
Cfg.GPUID = -1
# Setting the default value of the current time.
Cfg.currentTime = -1

# Setting the default value of the directories.
# Setting the default value of the model directory.
Cfg.modelDir = os.path.join('./', 'Checkpoints')
# Setting the default value of the log directory.
Cfg.logDir = os.path.join('./', 'Logs')
# Setting the default value of the dataset directory.
Cfg.dataDir = os.path.join('./', 'Data')

# Setting the arguments' parser.
def argParse():
    # Getting the configurator.
    CFG = Cfg
    # Creating the arguments' parser.
    parser = argparse.ArgumentParser(description = 'Argument Parser')
    # Setting the argument.
    parser.add_argument('-vs', '--vocabSize', type = int, dest = 'vs', default = CFG.vs, help = 'Integer => [1000, Infinite)')
    parser.add_argument('-es', '--embeddingSize', type = int, dest = 'es', default = CFG.es, help = 'Integer => [100, Infinite)')
    parser.add_argument('-cs', '--classSize', type = int, dest = 'cs', default = CFG.cs, help = 'Integer => [1, Infinite)')
    parser.add_argument('-lr', '--learningRate', type = float, dest = 'lr', default = CFG.lr, help = 'Float => [0, 1]')
    parser.add_argument('-beta1', '--beta1', type  = float, dest = 'beta1', default = CFG.beta1, help = 'Float => [0, 1]')
    parser.add_argument('-beta2', '--beta2', type = float, dest = 'beta2', default = CFG.beta2, help = 'Float => [0, 1]')
    parser.add_argument('-wd', '--weightDecay', type = float, dest = 'wd', default = CFG.wd, help = 'Float => [0, 1]')
    parser.add_argument('-bs', '--batchSize', type = int, dest = 'bs', default = CFG.bs, help = 'Integer => [1, Infinite)')
    parser.add_argument('-ep', '--epoches', type = int, dest = 'epoches', default = CFG.epoches, help = 'Integer => [1, Infinite)')
    parser.add_argument('-seed', '--seed', type = int, dest = 'seed', default = CFG.seed, help = 'Integer => [0, Infinite)')
    parser.add_argument('-gpu', '--GPUID', type = int, dest = 'GPUID', default = CFG.GPUID, help = 'Integer => [0, Infinite)')
    parser.add_argument('-currentTime', '--currentTime', type = str, dest = 'currentTime', default = CFG.currentTime, help = 'Format => %Y-%m-%d-%H-%M-%S')
    parser.add_argument('-modelDir', '--modelDir', type = str, dest = 'modelDir', default = CFG.modelDir, help = 'String')
    parser.add_argument('-logDir', '--logDir', type = str, dest = 'logDir', default = CFG.logDir, help = 'String')
    parser.add_argument('-dataDir', '--dataDir', type = str, dest = 'dataDir', default = CFG.dataDir, help = 'String')
    # Parsing the argument.
    args = vars(parser.parse_args())
    # Updating the configurator.
    CFG.update(args)
    # Returning the configurator.
    return Config(CFG)

# Testing the configurator.
if __name__ == "__main__":
    # Printing the items in configurator.
    print(f'''
        Vocabulary Size:        {Cfg.vs}
        Embedding Size:         {Cfg.es}
        Class Size:             {Cfg.cs}
        Learning Rate:          {Cfg.lr}
        Adam Beta One:          {Cfg.beta1}
        Adam Beta Two:          {Cfg.beta2}
        Weight Decay:           {Cfg.wd}
        Batch Size:             {Cfg.bs}
        Epoches:                {Cfg.epoches}
        Random Seed:            {Cfg.seed}
        GPU ID:                 {Cfg.GPUID}
        Model Directory:        {Cfg.modelDir}
        Log Directory:          {Cfg.logDir}
        Dataset Directory:      {Cfg.dataDir}
    ''')
    # Updating the configurator.
    Cfg = argParse()
    # Printing the items in configurator.
    print(f'''
        Vocabulary Size:        {Cfg.vs}
        Embedding Size:         {Cfg.es}
        Class Size:             {Cfg.cs}
        Learning Rate:          {Cfg.lr}
        Adam Beta One:          {Cfg.beta1}
        Adam Beta Two:          {Cfg.beta2}
        Weight Decay:           {Cfg.wd}
        Batch Size:             {Cfg.bs}
        Epoches:                {Cfg.epoches}
        Random Seed:            {Cfg.seed}
        GPU ID:                 {Cfg.GPUID}
        Model Directory:        {Cfg.modelDir}
        Log Directory:          {Cfg.logDir}
        Dataset Directory:      {Cfg.dataDir}
    ''')