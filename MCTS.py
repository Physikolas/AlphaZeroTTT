# An implementation of an MCTS class that is used in the AlphaZero algorithm. Mostly based on
# https://ufal.mff.cuni.cz/~straka/courses/npfl122/2021/slides.pdf/npfl122-2021-10.pdf,
# #https://suragnair.github.io/posts/alphazero.html and
#https://medium.com/@_michelangelo_/monte-carlo-tree-search-mcts-algorithm-for-dummies-74b2bae53bfa for how
# the initial MCTS algorithm works and how it could be implemented
import torch
import math
import numpy as np

class Node:
    def __init__(self, game, parameters, position, parent=None, selectedMoveFromParent=None, probabilityFromParent=0):
        #each node represents a position in the game. parameters are the parameters used later for the ALphaZero
        #method. position is said position of the game. parent is the node representing the position prior
        #to the current one, probabilityFromParent is the probability that in the prior position, the move leading
        # to the current situation (which is selectedMovefromParent) is taken
        self.game = game
        self.parameters = parameters
        self.position = position
        self.parent = parent
        self.selectedMoveFromParent = selectedMoveFromParent
        self.probabilityFromParent = probabilityFromParent
        self.children = [] #saving the child positions
        self.numVisits = 0 #numbers important for the AlphaZero/MCTS algo
        self.totalValue = 0 #numbers important for the AlphaZero/MCTS algo

    def isExpanded(self):
        #when there happened no expansion just yet in a certain position, this returns False.
        #If there was any expansion, the all possible moves are immediately explored by the MCTS/AlphaZero algo
        return len(self.children) > 0

    def selectSuccessor(self):
        #in an expansion phase of the MCTS, the child position with the highest UCB score is chosen
        successor = None
        ucbMAX = float("-inf")
        for child in self.children:
            UCB = self.getUCB(child) #see below
            if UCB > ucbMAX:
                ucbMAX = UCB
                successor = child
        return successor

    def getUCB(self, child):
        if child.numVisits == 0:
            return self.parameters['C'] * child.probabilityFromParent * (math.sqrt(self.numVisits) / (child.numVisits + 1))
        else:
            return 1 - ((child.totalValue / child.numVisits) + 1) / 2 + self.parameters['C'] * child.probabilityFromParent * (math.sqrt(self.numVisits) / (child.numVisits + 1))

    def backpropagate(self, value):
        #updates the visit counts and total value of all positions, recursively, that came from a certain
        # expansion and ended at a position with a value 'value'
        self.totalValue += value
        self.numVisits += 1
        value = -value #If a position is of value 'value' for Player 1, then the same position is of value
        #-'value' for player -1. Hence the change in sign
        if self.parent is not None:
            self.parent.backpropagate(value)

    def expand(self, moveDistribution):
        #expands in all possible directions, with support of a given distribution of move probabilities.
        #this will be provided by the NeuralNetwork
        legalMoves=self.game.getLegalMoves(self.position)
        for move in range(len(moveDistribution)):
            if legalMoves[move]==1:
                probability = moveDistribution[move]
                childPosition = self.position.copy()
                childPosition = -1 * self.game.getNextPosition(childPosition, move, 1)
                #the positions will always be encoded in such a way that the pieces set by the moving player
                #correspond to the '1's in the matrix, the others by -1
                child = Node(self.game, self.parameters, childPosition, self, move, probability)
                self.children.append(child)


class MCTS:
    def __init__(self, game, parameters, model):
        self.game = game
        self.parameters = parameters
        self.model = model

    @torch.no_grad()
    #by default, torch saves all the gradients when forward propagation is performed. But this
    #is not necessary here when the model is used only for evaluation. So by using this wrapper,
    #the program gets more efficient
    def explore(self, position):
        newNode=Node(self.game, self.parameters, position)
        newNode.numVisits=1
        initialExpansionProbabilities, value = self.model(
            torch.tensor(self.game.getEncodedPosition(position)).unsqueeze(0)
        )
        initialExpansionProbabilities=torch.softmax(initialExpansionProbabilities, axis=1).squeeze(0).cpu().numpy()
        initialExpansionProbabilities = ((1 - self.parameters['epsilon']) * initialExpansionProbabilities +
                                         self.parameters['epsilon'] * np.random.dirichlet([0.25] * self.game.numMoves))
        #see page 12 of https://ufal.mff.cuni.cz/~straka/courses/npfl122/2021/slides.pdf/npfl122-2021-10.pdf
        legalMoves = self.game.getLegalMoves(position)
        initialExpansionProbabilities = legalMoves*initialExpansionProbabilities
        initialExpansionProbabilities/= np.sum(initialExpansionProbabilities)
        newNode.expand(initialExpansionProbabilities)
        for indexExploration in range(self.parameters['numMCTSIterations']):
            node = newNode
            while node.isExpanded():
                node = node.selectSuccessor()
            value, gameFinished = self.game.getValueIfTerminated(node.position, node.selectedMoveFromParent)
            value = -value#again, value 1 from the viewpoint of player 1 is value -1 from the other player's viewpoint
            if not gameFinished:
                #if the position is not a final position, the value used for backpropagation should come from
                #the evaluation of the NN. otherwise, it should be 1 or -1, depending on the outcome of the match
                moveProbabilities, value = self.model(
                    torch.tensor(self.game.getEncodedPosition(node.position)).unsqueeze(0)
                    # in order to give the position to the NN, we have to 'encode it' in the form of an RGB picture
                    # by design, and then we have to perform an unsqueeze since we only want to evluate one position,
                    # not a batch of positions. Usually, the torch models are designed in such a way that a batch
                    # of data is given, that is, a 'list' of positions. The unsqueeze turns the position into a list
                    # [position] with one single element, making it compatible with the model's domain of definition
                )
                moveProbabilities = torch.softmax(moveProbabilities, axis=1).squeeze(0).cpu().numpy()
                legalMoves = self.game.getLegalMoves(node.position)
                #we need to further adjust the distribution for legal/illegal moves
                moveProbabilities = (moveProbabilities * legalMoves)/np.sum(moveProbabilities * legalMoves)
                node.expand(moveProbabilities)
                value = value.item()

            node.backpropagate(value)
        #now calculate the moveProbabilities, that is, the relative amount of times a move was chosen at the position
        moveProbabilities = np.zeros(self.game.numMoves)
        for child in newNode.children:
            moveProbabilities[child.selectedMoveFromParent] = child.numVisits
        moveProbabilities /= np.sum(moveProbabilities)
        return moveProbabilities