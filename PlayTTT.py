import numpy as np
import torch
import NN
import torch.nn as nn
import torch.nn.functional as F
from AZMCTS import AlphaZero
from AZMCTS import alphazeroNN
from TicTacToe import TicTacToe
from MCTS import MCTS
import matplotlib.pyplot as plt

TTT=TicTacToe(5,5,4)
position=np.zeros((5,5))
model = alphazeroNN(TTT,128)
model.load_state_dict(torch.load('updatedmodelTTT(5,5)_5.pt'))
parameters = {
    'C': 1.25,
    'numMCTSIterations': 600,
    'epsilon': 0.0 #for playing, we clearly do not want random sampling of moves
}
mcts = MCTS(TTT, parameters, model)
model.eval()
while(True):
    print(position)
    move=input('Make a move: ')
    try:
        position=TTT.getNextPosition(position,int(move),1)
    except ValueError:
        continue
    position=-position
    encPosition=TTT.getEncodedPosition(position)
    mctsProbabilities = mcts.explore(position)
    move = np.argmax(mctsProbabilities)
    print(f"Move chosen: {move}")
    position=TTT.getNextPosition(position,move,1)
    position=-position