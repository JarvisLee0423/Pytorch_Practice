#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/24
#   File Name:      Trainer.py
#   Description:    This file is used to training the model.
#============================================================================================#

# Importing the necessary library.
import numpy as np
import logging
import time
import os
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from visdom import Visdom
from Utils.Config import argParse
from Utils.DataPreprocessor import trainComponentsGenerator
from Model.Seq2SeqGRUModel import Encoder, Decoder, Seq2SeqGRUModelNN

# Getting the configurator.
Cfg = argParse()

# Creating the model directory.
if not os.path.exists(Cfg.modelDir):
    os.mkdir(Cfg.modelDir)
# Creating the log directory.
if not os.path.exists(Cfg.logDir):
    os.mkdir(Cfg.logDir)

# Setting the current time.
if Cfg.currentTime != -1:
    currentTime = Cfg.currentTime
else:
    currentTime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

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
    # Fixing the random seed.
    np.random.seed(Cfg.seed)
    torch.manual_seed(Cfg.seed)

# Creating the function for evaluation.
def evaluator(enDevSet, cnDevSet, enDevLen, cnDevLen, model, loss):
    # Setting the list for storing the evaluating loss and accuracy.
    evalLosses = []
    evalAcces = []
    # Evaluating the model.
    for i, batch in enumerate(enDevSet):
        # Getting the evaluating data.
        enBatch = batch.to(device)
        enLength = enDevLen[i]
        cnBatchIn = cnDevSet[i][:, :-1].to(device)
        cnBatchOut = cnDevSet[i][:, 1:].to(device)
        cnLength = [j - 1 for j in cnDevLen[i]]
        # Evaluting the model.
        prediction = model(enBatch, enLength, cnBatchIn, cnLength)
        # Computing the loss.
        cost = loss(prediction, cnBatchOut.reshape(-1))
        # Storing the loss.
        evalLosses.append(cost.item())
        # Computing the accuracy.
        accuracy = (torch.argmax(prediction, 1) == cnBatchOut.reshape(-1))
        accuracy = accuracy.sum().float() / len(accuracy)
        # Storing the accuracy.
        evalAcces.append(accuracy.item())
    # Returning the evaluating result.
    return np.sum(evalLosses) / len(evalLosses), np.sum(evalAcces) / len(evalAcces)

# Creating the training method.
def trainer(enTrainSet, cnTrainSet, enTrainLen, cnTrainLen, enDevSet, cnDevSet, enDevLen, cnDevLen, enVocabSize, cnVocabSize):
    # Creating the logging.
    logging.basicConfig(filename = Cfg.logDir + f'/logging-{currentTime}.txt', filemode = 'a', level = logging.INFO, format = '%(asctime)s %(levelname)s %(message)s', datefmt = '%Y-%m-%d %H:%M:%S %p')
    # Logging the information.
    logging.info(f'''
        Vocabulary Size:        {Cfg.vs}
        Hidden Size:            {Cfg.hs}
        Learning Rate:          {Cfg.lr}
        Adam Beta One:          {Cfg.beta1}
        Adam Beta Two:          {Cfg.beta2}
        Batch Size:             {Cfg.bs}
        Epoches:                {Cfg.epoches}
        Random Seed:            {Cfg.seed}
        GPU ID:                 {Cfg.GPUID}
        Model Directory:        {Cfg.modelDir}
        Log Directory:          {Cfg.logDir}
        Dataset Directory:      {Cfg.dataDir}
    ''')
    # Creating the visdom.
    vis = Visdom(env = 'Seq2SeqGRUModel')
    # Creating the graph.
    lossGraph = vis.line(X = [0], Y = [0], opts = dict(legend = ['TrainingLoss', 'EvaluatingLoss'], xlabel = 'Epoches', ylabel = 'Loss', title = f'Training and Evaluating Loss - {currentTime}'), name = 'TrainingLoss')
    vis.line(X = [0], Y = [0], win = lossGraph, update = 'append', name = 'EvaluatingLoss')
    accGraph = vis.line(X = [0], Y = [0], opts = dict(legend = ['TrainingAcc', 'EvaluatingAcc'], xlabel = 'Epoches', ylabel = 'Acc', title = f'Training and Evaluating Acc - {currentTime}'), name = 'TrainingAcc')
    vis.line(X = [0], Y = [0], win = accGraph, update = 'append', name = 'EvaluatingAcc')
    # Creating the encoder.
    encoder = Encoder(enVocabSize, Cfg.hs).to(device)
    # Creating the decoder.
    decoder = Decoder(cnVocabSize, Cfg.hs).to(device)
    # Creating the sequence to sequence model.
    model = Seq2SeqGRUModelNN(encoder, decoder).to(device)
    # Creating the loss function.
    loss = nn.CrossEntropyLoss()
    # Creating the optimizer.
    optimizer = optim.Adam(model.parameters(), lr = Cfg.lr, betas = [Cfg.beta1, Cfg.beta2])
    # Setting the list to storing the training loss.
    trainLosses = []
    # Setting the list to storing the training accuracy.
    trainAccs = []
    # Setting the list to storing the evaluating loss.
    evalLosses = []
    # Setting the list to storing the evaluating accuracy.
    evalAccs = []
    # Training the model.
    for epoch in range(Cfg.epoches):
        # Setting the list for storing the training loss and accuracy.
        trainLoss = []
        trainAcc = []
        # Setting the loading bar.
        with tqdm(total = len(enTrainSet), desc = f'Epoch {epoch + 1}/{Cfg.epoches}', unit = 'batches', dynamic_ncols = True) as pbars:
            for i, bacth in enumerate(enTrainSet):
                # Getting the training data.
                enBatch = bacth.to(device)
                enLength = enTrainLen[i]
                # Decoder do not need the last words for inputting.
                cnBatchIn = cnTrainSet[i][:, :-1].to(device)
                # Decoder do not need the first word for predicting.
                cnBatchOut = cnTrainSet[i][:, 1:].to(device)
                cnLength = [j - 1 for j in cnTrainLen[i]]
                # Training the model.
                # Getting the prediction.
                prediction = model(enBatch, enLength, cnBatchIn, cnLength)
                # Computing the loss.
                cost = loss(prediction, cnBatchOut.reshape(-1))
                # Storing the cost.
                trainLoss.append(cost.item())
                # Clearing the gradient.
                optimizer.zero_grad()
                # Applying the backward propagation.
                cost.backward()
                # Updating the parameters.
                optimizer.step()
                # Computing the accuracy.
                accuracy = (torch.argmax(prediction, 1) == cnBatchOut.reshape(-1))
                accuracy = accuracy.sum().float() / len(accuracy)
                # Storing the accuracy.
                trainAcc.append(accuracy.item())

                # Updating the loading bar.
                pbars.update(1)
                # Updating the training information.
                pbars.set_postfix_str(' - Train Loss %.4f - Train Acc %.4f' % (np.mean(trainLoss), np.mean(trainAcc)))
        # Closing the loading bar.
        pbars.close()
        # Printing the hint for evaluating.
        print('Evaluating...', end = ' ')   
        # Evalutaing the model.
        evalLoss, evalAcc = evaluator(enDevSet, cnDevSet, enDevLen, cnDevLen, model.eval(), loss)
        # Printing the evaluating information.
        print(' - Eval Loss %.4f - Eval Acc %.4f' % (evalLoss, evalAcc))
        # Storing the training and evaluating information.
        trainLosses.append(np.mean(trainLoss))
        trainAccs.append(np.mean(trainAcc))
        evalLosses.append(evalLoss)
        evalAccs.append(evalAcc)
        # Logging the information.
        logging.info('Epoch [%d/%d] -> Training: Loss [%.4f] - Acc [%.4f] || Evaluating: Loss [%.4f] - Acc [%.4f]' % (epoch + 1, Cfg.epoches, np.mean(trainLoss), np.mean(trainAcc), evalLoss, evalAcc))
        # Drawing the graph.
        vis.line(
            X = [k for k in range(1, len(trainLosses) + 1)],
            Y = trainLosses,
            win = lossGraph,
            update = 'new',
            name = 'TrainingLoss'
        )
        vis.line(
            X = [k for k in range(1, len(evalLosses) + 1)],
            Y = evalLosses,
            win = lossGraph,
            update = 'new',
            name = 'EvaluatingLoss'
        )
        vis.line(
            X = [k for k in range(1, len(trainAccs) + 1)],
            Y = trainAccs,
            win = accGraph,
            update = 'new',
            name = 'TrainingAcc'
        )
        vis.line(
            X = [k for k in range(1, len(evalAccs) + 1)],
            Y = evalAccs,
            win = accGraph,
            update = 'new',
            name = 'EvaluatingAcc'
        )
        # Giving the hint for saving the model.
        logging.info("Model Saved")
        # Saving the model.
        try:
            torch.save(encoder.train().state_dict(), Cfg.modelDir + f'/Encoder-{currentTime}/Seq2SeqEncoder-Epoch{epoch + 1}.pt')
            torch.save(decoder.train().state_dict(), Cfg.modelDir + f'/Decoder-{currentTime}/Seq2SeqDecoder-Epoch{epoch + 1}.pt')
        except:
            os.mkdir(Cfg.modelDir + f'/Encoder-{currentTime}/')
            os.mkdir(Cfg.modelDir + f'/Decoder-{currentTime}/')
            torch.save(encoder.train().state_dict(), Cfg.modelDir + f'/Encoder-{currentTime}/Seq2SeqEncoder-Epoch{epoch + 1}.pt')
            torch.save(decoder.train().state_dict(), Cfg.modelDir + f'/Decoder-{currentTime}/Seq2SeqDecoder-Epoch{epoch + 1}.pt')
        # Converting the model state.
        model = model.train()
    # Saving the graph.
    vis.save(envs = ['Seq2SeqGRUModel'])

# Setting the main function.
if __name__ == "__main__":
    # Getting the training data.
    enItos, enStoi, cnItos, cnStoi, enTrainSet, cnTrainSet, enTrainLen, cnTrainLen = trainComponentsGenerator.trainComponentsGenerator(Cfg.dataDir, Cfg.bs, Cfg.vs, shuffle = True, train = True)
    # Getting the development data.
    _, _, _, _, enDevSet, cnDevSet, enDevLen, cnDevLen = trainComponentsGenerator.trainComponentsGenerator(Cfg.dataDir, Cfg.bs, Cfg.vs, shuffle = False, train = False)
    # Checking the training data.
    for i in range(len(enTrainSet)):
        for k in range(enTrainSet[i].shape[0]):
            print(" ".join([enItos[index] for index in enTrainSet[i][k]]))
            print(" ".join([cnItos[index] for index in cnTrainSet[i][k]]))
            # Getting the command.
            cmd = input("'Exit' for quit looking the training data: ")
            # Handling the command.
            if cmd == 'Exit':
                break
        if cmd == 'Exit':
            break
    # Checking the development data.
    for i in range(len(enDevSet)):
        for k in range(enDevSet[i].shape[0]):
            print(" ".join([enItos[index] for index in enDevSet[i][k]]))
            print(" ".join([cnItos[index] for index in cnDevSet[i][k]]))
            # Getting the command.
            cmd = input("'Exit' for quit looking the development data: ")
            # Handling the command.
            if cmd == 'Exit':
                break
        if cmd == 'Exit':
            break
    # Getting the input command.
    cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
    # Handling the command.
    while cmd != 'Exit':
        if cmd == 'T':
            # Training the model.
            trainer(enTrainSet, cnTrainSet, enTrainLen, cnTrainLen, enDevSet, cnDevSet, enDevLen, cnDevLen, len(enItos), len(cnItos))
            # Getting the command.
            cmd = input("Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")
        elif cmd == 'E':
            # Checking whether there is the model or not.
            try:
                # Creating the model.
                encoder = Encoder(len(enItos), hiddenSize)
                decoder = Decoder(len(cnItos), hiddenSize)
                # Loading the model.
                encoder.load_state_dict(torch.load(Cfg.modelDir + f'/Encoder-{currentTime}/Seq2SeqEncoder-Epoch{Cfg.epoches}.pt'))
                decoder.load_state_dict(torch.load(Cfg.modelDir + f'/Decoder-{currentTime}/Seq2SeqDecoder-Epoch{Cfg.epoches}.pt'))
                # Making the model into evaluating state.
                encoder = encoder.to(device).eval()
                decoder = decoder.to(device).eval()
                # Getting the input English sentence.
                sentence = input("Please input the English sentence ('Exit' for quit): ")
                # Handling the sentence.
                while sentence != 'Exit':
                    # Setting the list to storing the translation.
                    translation = []
                    # Initializing the index.
                    index = 2
                    # Spliting the sentence.
                    enData = ['<bos>'] + nltk.word_tokenize(sentence) + ['<eos>']
                    # Getting the length of the input English data.
                    length = [len(enData)]
                    # Getting the evaluating data.
                    enData = torch.LongTensor(np.array([enStoi.get(word, enStoi.get('<unk>')) for word in enData]).astype('int64')).unsqueeze(0).to(device)
                    # Getting the context hidden.
                    hidden = encoder(enData, length)
                    # Getting the first trainslated word.
                    prediction = torch.LongTensor(np.array([cnStoi['<bos>']]).astype('int64')).unsqueeze(0).to(device)
                    # Setting the counter to limit the length of the translated sentence.
                    count = 0
                    # Getting the trainslation.
                    while cnItos[index] != '<eos>':
                        # Computing the counter.
                        count += 1
                        # Getting the prediction.
                        prediction, hidden = decoder(prediction, [1], hidden)
                        # Getting the index.
                        index = torch.argmax(prediction, 1)
                        # Getting the word.
                        if cnItos[index] != '<eos>':
                            translation.append(cnItos[index])
                        # Reseting the prediction.
                        prediction = torch.LongTensor(np.array([index]).astype('int64')).unsqueeze(0).to(device)
                        # Distinguishing whether the length is long enough.
                        if count >= 20:
                            break
                    # Printing the tanslation.
                    print("The Chinese translation is: " + " ".join(translation))
                    # Getting the input English sentence.
                    sentence = input("Please input the English sentence ('Exit' for quit): ")
                # Quiting the system.
                cmd = sentence
            except:
                # Giving the hint.
                print("There is no model! Please training one first!")
                # Training the model.
                cmd = 'T'
        else:
            # Giving the hint.
            cmd = input("Invalid command! Please input the command ('T' for training, 'E' for evaluating, 'Exit' for quit): ")