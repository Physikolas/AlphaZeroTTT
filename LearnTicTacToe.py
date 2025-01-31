import numpy as np
import torch
import NN
import torch.nn as nn
import torch.nn.functional as F
from AZMCTS import AlphaZero
from NN import alphazeroNN
from TicTacToe import TicTacToe

TTT = TicTacToe(5,5,4)

model = alphazeroNN(TTT, 128) #for the (5,5)-model

optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)#standard parameters for adams
model.load_state_dict(torch.load('updatedmodelTTT(5,5)_5.pt'))
optimizer.load_state_dict(torch.load('updatedoptimizerTTT(5,5)_5.pt'))
parameters = {
    'C': 2,
    'numAlphaZeroIterations': 4,
    'numMCTSIterations': 500,
    'batchSize': 256,
    'numSelfPlayIterations': 1000,
    'numEpochs': 3,
    'epsilon': 0.3 #for training, some random sampling in the move choice makes sense to learn new positions
}
alphaZero = AlphaZero(model,TTT, parameters, optimizer)
alphaZero.learn() #saves (overwrites) the new weights as 'updatedmodelTTT(5,5)_{iteration}.pt', where iteration
# reaches from 0 to 3 if 'numAlphaZeroIterations'=4, for example