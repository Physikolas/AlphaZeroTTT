import numpy as np
from MCTS import MCTS
import torch.nn
import torch.nn.functional as F
import random

# See https://www.chessprogramming.org/AlphaZero#Network_Architecture for details on the architecture
class alphazeroNN(torch.nn.Module):
    # alphaZero is, in its essence, a NN for image recognition. That is, it is primarily
    # a convolutional NN, where the 'current game state' is encoded as an RGB image of size 3x3 (for TicTacToe).
    # The 'R'-part of the image is represented by the pieces of player 1, the 'G'-part represented by the pieces
    #set by player 2, and the B'-part represented by empty spots on the board
    def __init__(self, game, numFilters): #numFilters is the number of filters in the convNetwork
        super().__init__()
        self.startBlock = torch.nn.Sequential(
            torch.nn.Conv2d(3, numFilters, kernel_size=3, padding=1), #
            torch.nn.BatchNorm2d(numFilters),
            torch.nn.ReLU()
        )
        #the startblock consists of purely one convolutional layer, takes in the three channels 'R', 'G' and 'B'
        # and outputs the convolutions. It is standard to normalize the outputs with a batchnorm layer and apply a ReLU
        #function to improve the model

        self.resBlock1 = ResBlock(numFilters)
        self.resBlock2 = ResBlock(numFilters)
        self.resBlock3 = ResBlock(numFilters)
        self.resBlock4 = ResBlock(numFilters)
#look below for the definition of a resblock. In principle, the number of these blocks can be anything, but should
        #represent the complexity of the game. The chess engine had 19 of them

        self.outputMoveProbabilities = torch.nn.Sequential(
            # this be understood as the prediciton of the model, to which likelihood a certain action is best
            torch.nn.Conv2d(numFilters, 32, kernel_size=3, padding=1),#the number 32 for the outchannels has no
            # deeper meaning. One usually takes a power of 2, and 32 seems reasonable in terms of complexity of the game
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * game.numRows * game.numColumns, game.numMoves),
        )

        self.outputValue = torch.nn.Sequential(
            # this can be understood as the evaluation of the model for a given state
            torch.nn.Conv2d(numFilters, 3, kernel_size=3, padding=1),#again, the number 3 for theoutchannels has no
            # deeper meaning. It makes sense, though, to take a value similar to that for the outblockMove, after adjusting for
            #the fact that the output here consists of a single value as opposed to an array of 7, for example, real values
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3 * game.numRows * game.numColumns, 1),
            torch.nn.Tanh() #this is the standard choice if values between -1 and 1 are demanded. This has some nice properties,
            #for example that the derivative near 0 changes 'slowly'
        )

    def forward(self, x):
        x = self.startBlock(x)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.resBlock4(x)
        moveProbabilities = self.outputMoveProbabilities(x)
        value = self.outputValue(x)
        return moveProbabilities, value


class ResBlock(torch.nn.Module):
    #in principle, a resblock is simply the concatenation of two conv layers, but the forwarding process is altered
    # by adding the input to the output of the two layers (see 'def forward')
    def __init__(self, numFilters):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(numFilters, numFilters, kernel_size=3, padding=1)
        self.bn2D1 = torch.nn.BatchNorm2d(numFilters)
        self.conv2 = torch.nn.Conv2d(numFilters, numFilters, kernel_size=3, padding=1)
        self.bn2D2 = torch.nn.BatchNorm2d(numFilters)

    def forward(self, x):
        res = x
        x = F.relu(self.bn2D1(self.conv1(x)))
        x = self.bn2D2(self.conv2(x))
        x += res
        x = F.relu(x)
        return x

class AlphaZero:
    def __init__(self, model, game, parameters, optimizer): #to parameters: is given as a dictionary
        # ['C', numAlphaZeroIterations, numMCTSIterations, numBatches, numSelfPlayIterations, numEpochs]
        # it is assumed that numSelfPlayIterations (how often alphazero plays against itself)
        # is an integer multiple of numBatches, the number of batches taken for the backpropagation optimization method
        # of the NN. the other parameters:
        # 'C' is some integer >1 (2, for example) that occurs in the UCB value.
        # numAlphaZeroIterations is the number of times AlphaZero undergoes the whole learning process.
        # numMCTSIterations is the amount of new positions that the MCTS discovers during one expansion phase
        # numEpochs determines how many backpropagation cycles the neural network goes through under a given set of
        #self-played games
        self.parameters = parameters
        self.model = model
        self.game = game
        self.optimizer = optimizer
        self.mcts = MCTS(game, parameters, model)

    #during self play, the algorithm decides, using a MCTS with help of the engine, which move is next
    def selfPlay(self):
        player = 1
        workingMemory = [] #used only to remember positions/move probabilities/values for one game simulation
        position = np.zeros((self.game.numRows, self.game.numColumns))
        gameHasFinished=False
        while not gameHasFinished:
            changedPerspective = position * player
            moveProbabilities = self.mcts.explore(position)
            workingMemory.append((changedPerspective, moveProbabilities, player))
            move = np.random.choice(self.game.numMoves, p=moveProbabilities)
            position = self.game.getNextPosition(position, move, player)
            value, gameHasFinished = self.game.getValueIfTerminated(position, move)
            returnedWorkingMemory = [] #to save the positions, the observed Move probabilities and the respective
            #outcomes that have been observed
            if gameHasFinished:
                for savedPosition, savedMoveProbabilities, savedPlayer in workingMemory:
                    outcome = (savedPlayer * player) * value
                    returnedWorkingMemory.append((self.game.getEncodedPosition(savedPosition),savedMoveProbabilities,outcome))
                return returnedWorkingMemory
            player = -player

    def train(self, memory):
        random.shuffle(memory)
        for batchNumber in range(0, len(memory), self.parameters['batchSize']):
            sample = memory[batchNumber:min(len(memory) - 1, batchNumber + self.parameters['batchSize'])]
            #to prevent an indexOutOfBoundError, it is unclear how long memory will be since a single match has an
            #undetermined number of moves
            observedPosition, observedMoveProbabilities, observedValues = zip(*sample)# so that
            #there are 3 separate tupels for all observed positions, probs and values each as opposed to
            #multiple tupels containing precisely one position, probs and value

            observedPosition= np.array(observedPosition)
            observedMoveProbabilities=np.array(observedMoveProbabilities)
            observedValues=np.array(observedValues).reshape(-1, 1) ## it is important for later that each value
            #has its own subarray, which it does not have after the zip command above

            observedPosition = torch.tensor(observedPosition, dtype=torch.float32)
            observedMoveProbabilities = torch.tensor(observedMoveProbabilities, dtype=torch.float32)
            observedValues = torch.tensor(observedValues, dtype=torch.float32)#this is where it is important that
            #each value has its own subarray

            predictedMoveProbabilities, predictedValue = self.model(observedPosition)

            moveProbabilityLoss = F.cross_entropy(predictedMoveProbabilities, observedMoveProbabilities)
            valueLoss = F.mse_loss(predictedValue, observedValues)
            totalLoss = moveProbabilityLoss + valueLoss

            self.optimizer.zero_grad() #https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            totalLoss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.parameters['numAlphaZeroIterations']):
            memory = [] #the memory is discarded after each iteration. The idea is that the games played after getting
            #an update to the NN are better suited for further learning. This is certainly true at the beginning,
            # though it should be said that at some point, the engine is so good that it is worth to save games/positions
            # played earlier. AlphaZero for chess has a huge database of positions together with evaluations etc
            self.model.eval()
            for selfPlayIteration in range(self.parameters['numSelfPlayIterations']):
                memory += self.selfPlay()
                print(selfPlayIteration)
                print(len(memory))

            self.model.train()
            for epoch in range(self.parameters['numEpochs']):
                self.train(memory)

            torch.save(self.optimizer.state_dict(), f"updatedoptimizerTTT(5,5)_{iteration}.pt")
            torch.save(self.model.state_dict(), f"updatedmodelTTT(5,5)_{iteration}.pt")