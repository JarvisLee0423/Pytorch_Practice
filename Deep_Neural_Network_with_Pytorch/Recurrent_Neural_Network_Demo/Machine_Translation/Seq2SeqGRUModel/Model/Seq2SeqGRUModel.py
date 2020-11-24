#============================================================================================#
#   Copyright:          JarvisLee                       
#   Date:               2020/11/24
#   Project Name:       Seq2SeqGRUModel.py
#   Description:        Build the sequence to sequence model to handle the English to Chinese
#                       machine translation problem.
#   Model Description:  Input               ->  Inputting English sentence
#                       Encoder             ->  Computing the context vector
#                       Decoder             ->  Getting the translated Chinese sentence
#============================================================================================#

# Importing the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Creating the class for encoder.
class Encoder(nn.Module):
    # Creating the constructor.
    def __init__(self, vocabSize, hiddenSize):
        # Inheriting the super constructor.
        super(Encoder, self).__init__()
        # Creating the model.
        self.encoderEmbed = nn.Embedding(vocabSize, hiddenSize)
        self.encoderGRU = nn.GRU(hiddenSize, hiddenSize, batch_first = True)
        self.encoderDropout = nn.Dropout(0.2)
    # Creating the forward function.
    def forward(self, enData, enLength):
        # Applying the embedding layer. [batchSize, sequenceLength] -> [batchSize, sequenceLength, hiddenSize]
        embed = self.encoderEmbed(enData)
        # Applying the dropout. [batchSize, sequenceLength, hiddenSize] -> [batchSize, sequenceLength, hiddenSize]
        embed = self.encoderDropout(embed)
        # Unpadding the data. [batchSize, sequenceLength, hiddenSize] -> [batchSize, sentenceLength, hiddenSize]
        packedEmbed = nn.utils.rnn.pack_padded_sequence(embed, enLength, batch_first = True, enforce_sorted = False)
        # Applying the GRU. [batchSize, sentenceLength, hiddenSize] -> [1, batchSize, hiddenSize]
        _, hidden = self.encoderGRU(packedEmbed)
        # Returning the hidden.
        return hidden

# Creating the class for decoder.
class Decoder(nn.Module):
    # Creating the constructor.
    def __init__(self, vocabSize, hiddenSize):
        # Inheriting the super constructor.
        super(Decoder, self).__init__()
        # Creating the model.
        self.decoderEmbed = nn.Embedding(vocabSize, hiddenSize)
        self.decoderGRU = nn.GRU(hiddenSize, hiddenSize, batch_first = True)
        self.decoderLinear = nn.Linear(hiddenSize, vocabSize)
        self.decoderDropout = nn.Dropout(0.2)
    # Creating the forward function.
    def forward(self, cnData, cnLength, hidden):
        # Applying the embedding layer. [batchSize, sequenceLength] -> [batchSize, sequenceLength, hiddenSize]
        embed = self.decoderEmbed(cnData)
        # Applying the dropout. [batchSize, sequenceLength, hiddenSize] -> [batchSize, sequenceLength, hiddenSize]
        embed = self.decoderDropout(embed)
        # Unpadding the data. [batchSize, sequenceLength, hiddenSize] -> [batchSize, sentenceLength, hiddenSize]
        packedEmbed = nn.utils.rnn.pack_padded_sequence(embed, cnLength, batch_first = True, enforce_sorted = False)
        # Applying the GRU. [batchSize, sentenceLength, hiddenSize] -> [batchSize, sentenceLength, hiddenSize]
        packedOutput, hidden = self.decoderGRU(packedEmbed, hidden)
        # Unpacking the data. [batchSize, sentenceLength, hiddenSize] -> [batchSize, sequenceLength, hiddenSize]
        output, _ = nn.utils.rnn.pad_packed_sequence(packedOutput, batch_first = True)
        # Reshaping the output. [batchSize, sequenceLength, hiddenSize] -> [batchSize * sequenceLength, hiddenSize]
        output = output.reshape(-1, output.shape[2])
        # Applying the linear. [batchSize * sequenceLength, hiddenSize] -> [batchSize * sequenceLength]
        output = self.decoderLinear(output)
        # Returning the output.
        return output, hidden

# Creating the Sequence to Sequence model.
class Seq2SeqGRUModelNN(nn.Module):
    # Creating the constructor.
    def __init__(self, encoder, decoder):
        # Inheriting the super constructor.
        super(Seq2SeqGRUModelNN, self).__init__()
        # Creating the model.
        self.encoder = encoder
        self.decoder = decoder
    # Creating the forward function.
    def forward(self, enData, enLength, cnData, cnLength):
        # Applying the encoder.
        hidden = self.encoder(enData, enLength)
        # Applying the decoder.
        output, _ = self.decoder(cnData, cnLength, hidden)
        # Returning the output.
        return output