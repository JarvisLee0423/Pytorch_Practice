#============================================================================================#
#   Copyright:      JarvisLee
#   Date:           2020/11/24
#   File Name:      DataPreprocessor.py
#   Description:    This file is used to setting the data preprocessor components.
#============================================================================================#

# Importing the necessary library.
import numpy as np
import torch
import nltk
import jieba
from collections import Counter

# Creating the class for generating the necessary training component.
class trainComponentsGenerator():
    # Setting the method to read the training data.
    @staticmethod
    def dataReader(root, train = True):
        # Getting the corresponding data.
        if train:
            # Opening the file.
            trainFile = open(root + '/train.txt', 'r', encoding = 'utf-8')
            # Getting all the sentences.
            sentences = trainFile.readlines()
            # Setting the list to storing the sentences.
            enSentence = []
            cnSentence = []
            # Getting all the sentence.
            for each in sentences:
                # Saperating the English sentence and Chinese sentence.
                temp = each.split('\t')
                # Getting the English sentence and Chinese sentence.
                enSentence.append(temp[0])
                cnSentence.append(temp[1].split('\n')[0])
        else:
            # Opening the file.
            devFile = open(root + '/dev.txt', 'r', encoding = 'utf-8')
            # Getting all the sentences.
            sentences = devFile.readlines()
            # Setting the list to storing the sentences.
            enSentence = []
            cnSentence = []
            # Getting all the sentence.
            for each in sentences:
                # Saperating the English sentence and Chinese sentence.
                temp = each.split('\t')
                # Getting the English sentence and Chinese sentence.
                enSentence.append(temp[0])
                cnSentence.append(temp[1].split('\n')[0])
        # Returning the data.
        return enSentence, cnSentence
    # Setting the method to tokenize the sentence.
    @staticmethod
    def tokenizer(root, train = True):
        # Tokenizing the data.
        if train:
            # Getting the sentence.
            en, cn = trainComponentsGenerator.dataReader(root, train)
            # Setting the list to store the English data.
            enData = []
            # Spliting the English sentence.
            for sentence in en:
                # Getting the tokenized sentence.
                enData.append(['<bos>'] + nltk.word_tokenize(sentence) + ['<eos>'])
            # Setting the list to store the Chinese data.
            cnData = []
            # Spliting the Chinese sentence.
            for sentence in cn:
                # Getting the tokenized sentence.
                cnData.append(['<bos>'] + jieba.lcut(sentence) + ['<eos>'])
        else:
            # Getting the sentence.
            en, cn = trainComponentsGenerator.dataReader(root, train)
            # Setting the list to store the English data.
            enData = []
            # Spliting the English sentence.
            for sentence in en:
                # Getting the tokenized sentence.
                enData.append(['<bos>'] + nltk.word_tokenize(sentence) + ['<eos>'])
            # Setting the list to store the Chinese data.
            cnData = []
            # Spliting the Chinese sentence.
            for sentence in cn:
                # Getting the tokenized sentence.
                cnData.append(['<bos>'] + jieba.lcut(sentence) + ['<eos>'])
        # Returning the tokenized data.
        return enData, cnData
    # Setting the method to generating the vocabulary.
    @staticmethod
    def vocabGenerator(root, vocabSize):
        # Getting the tokenized data.
        en, cn = trainComponentsGenerator.tokenizer(root, True)
        # Setting the list to conut the English words' appearance frequency.
        enWordFreq = Counter()
        # Getting the English words' appearance frequency.
        for sentence in en:
            for word in sentence:
                # Checking whether the word is begin symbol or ending symbol.
                if word != '<bos>' and word != '<eos>':
                    enWordFreq[word] += 1
        # Getting the English vocabulary.
        enVocab = ['<unk>', '<pad>', '<bos>', '<eos>']
        for word in enWordFreq.most_common(vocabSize):
            enVocab.append(word[0])
        # Forming the word to index tool.
        enStoi = {}
        for each in enVocab:
            enStoi[each] = enVocab.index(each)
        # Setting the list to count the Chinese words' appearance frequency.
        cnWordFreq = Counter()
        # Getting the Chinese words' appearance frequency.
        for sentence in cn:
            for word in sentence:
                # Checking whether the word is begin symbol or ending symbol.
                if word != '<bos>' and '<eos>':
                    cnWordFreq[word] += 1
        # Getting the Chinese vocabulary.
        cnVocab = ['<unk>', '<pad>', '<bos>', '<eos>']
        for word in cnWordFreq.most_common(vocabSize):
            cnVocab.append(word[0])
        # Forming the word to index tool.
        cnStoi = {}
        for each in cnVocab:
            cnStoi[each] = cnVocab.index(each)
        # Returning the tools
        return enVocab, enStoi, cnVocab, cnStoi
    # Setting the method to seperate the minibatch.
    @staticmethod
    def minibacthGenerator(root, batchSize, shuffle = True, train = True):
        # Getting the data.
        en, _ = trainComponentsGenerator.tokenizer(root, train)
        # Getting the minibacth.
        minibatches = []
        # Getting the minibatches.
        for i in range(0, len(en), batchSize):
            minibatches.append(i)
        # Shuffle the minibatches.
        if shuffle:
            np.random.shuffle(minibatches)
        # Returning the minibatches
        return minibatches
    # Setting the method to forming the minibatch.
    @staticmethod
    def trainComponentsGenerator(root, batchSize, vocabSize, shuffle = True, train = True):
        # Getting the minibatches.
        minibatches = trainComponentsGenerator.minibacthGenerator(root, batchSize, shuffle, train)
        # Getting the tokenized data.
        en, cn = trainComponentsGenerator.tokenizer(root, train)
        # Getting the vocabulary tools.
        enItos, enStoi, cnItos, cnStoi = trainComponentsGenerator.vocabGenerator(root, vocabSize)
        # Setting the list to store the minibatch.
        enMinibatches, cnMinibatches, enLengths, cnLengths = [], [], [], []
        # Forming the minibatch.
        for i in minibatches:
            # Checking whehter the last minibatch is large enough.
            if (i + batchSize - 1) <= (len(en) - 1):
                # Getting the raw minibatch.
                enRawMinibatch = en[i : i + batchSize]
                cnRawMinibatch = cn[i : i + batchSize]
                enLength = [len(sent) for sent in enRawMinibatch]
                cnLength = [len(sent) for sent in cnRawMinibatch]
                # Getting the max length of English minibatch and Chinese minibatch.
                enMaxLength = max(enLength)
                cnMaxLength = max(cnLength)
                # Padding the English minibatch and digitalizing the minibatch.
                for j, sent in enumerate(enRawMinibatch):
                    # Checking whether the sentence is long enough.
                    while len(sent) < enMaxLength:
                        sent.append('<pad>')
                    # Digitalizing the minibatch.
                    sent = [enStoi.get(word, enStoi.get('<unk>')) for word in sent]
                    enRawMinibatch[j] = sent
                # Padding the Chinese minibatch and digitalizing the minibatch.
                for j, sent in enumerate(cnRawMinibatch):
                    # Checking whether the sentence is long enough.
                    while len(sent) < cnMaxLength:
                        sent.append('<pad>')
                    # Digitalizing the minibatch.
                    sent = [cnStoi.get(word, cnStoi.get('<unk>')) for word in sent]
                    cnRawMinibatch[j] = sent
            else:
                # Getting the raw minibatch.
                enRawMinibatch = en[i : (len(en) - 1)]
                cnRawMinibatch = cn[i : (len(cn) - 1)]
                enLength = [len(sent) for sent in enRawMinibatch]
                cnLength = [len(sent) for sent in cnRawMinibatch]
                # Getting the max length of English minibatch and Chinese minibatch.
                enMaxLength = max(enLength)
                cnMaxLength = max(cnLength)
                # Padding the English minibatch and digitalizing the minibatch.
                for j, sent in enumerate(enRawMinibatch):
                    # Checking whether the sentence is long enough.
                    while len(sent) < enMaxLength:
                        sent.append('<pad>')
                    # Digitalizing the minibatch.
                    sent = [enStoi.get(word, enStoi.get('<unk>')) for word in sent]
                    enRawMinibatch[j] = sent
                # Padding the Chinese minibatch and digitalizing the minibatch.
                for j, sent in enumerate(cnRawMinibatch):
                    # Checking whether the sentence is long enough.
                    while len(sent) < cnMaxLength:
                        sent.append('<pad>')
                    # Digitalizing the minibatch.
                    sent = [cnStoi.get(word, cnStoi.get('<unk>')) for word in sent]
                    cnRawMinibatch[j] = sent
            # Converting the minibatch into the tensor.
            enMinibatch = torch.LongTensor(np.array(enRawMinibatch).astype('int64'))
            cnMinibatch = torch.LongTensor(np.array(cnRawMinibatch).astype('int64'))
            # Storing the minibatch.
            enMinibatches.append(enMinibatch)
            cnMinibatches.append(cnMinibatch)
            enLengths.append(enLength)
            cnLengths.append(cnLength)
        # Returning the train components.
        return enItos, enStoi, cnItos, cnStoi, enMinibatches, cnMinibatches, enLengths, cnLengths