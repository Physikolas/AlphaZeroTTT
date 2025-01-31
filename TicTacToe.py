import numpy as np

class TicTacToe:

    def __init__(self,numRows,numColumns,numToWin):
        self.numRows=numRows
        self.numColumns=numColumns
        self.numMoves=numRows*numColumns
        self.numToWin=numToWin

    def getLegalMoves(self,position):
        return np.reshape(position==0,-1).astype(int)

    def getNextPosition(self, position, move, player):
        row=move//self.numColumns
        column=move%self.numColumns
        position[row,column]=player
        return position

    def countPiecesInGivenDirection(self,position,move,rowDirection,columnDirection):
    #position is the position after the move has been made. counts successive pieces of the moving player, starting at
    #the position of the new piece
        row = move // self.numColumns
        column = move % self.numColumns
        player=position[row,column]
        for i in range(1,self.numToWin):
            nextRow=row+i*rowDirection
            nextColumn = column + i * columnDirection
            if(
                nextRow<0 or nextRow==self.numRows or
                nextColumn < 0 or nextColumn == self.numColumns or
                position[nextRow,nextColumn]!=player
            ):
                return i-1
        return self.numToWin-1

    def checkIfWin(self,position,move):
        #position is the position after the move has been made
        if move == None:
            return False
        row = move // self.numColumns
        column = move % self.numColumns
        return (
            ##check all possible directions
            self.countPiecesInGivenDirection(position,move,1,0)+
            self.countPiecesInGivenDirection(position,move,-1,0)>=self.numToWin-1 or #vertical direction
            self.countPiecesInGivenDirection(position,move, 0, 1)+
            self.countPiecesInGivenDirection(position,move, 0, -1)>= self.numToWin-1 or #horizontal direction
            self.countPiecesInGivenDirection(position, move, 1, 1) +
            self.countPiecesInGivenDirection(position, move, -1, -1) >= self.numToWin-1 or #the diagonal
            self.countPiecesInGivenDirection(position, move, -1, 1) +
            self.countPiecesInGivenDirection(position, move, 1, -1) >= self.numToWin-1 #the anti-diagonal
        )


    def getValueIfTerminated(self,position,move):
        if np.sum(self.getLegalMoves(position))==0:
            return 0,True
        if self.checkIfWin(position,move):
            return 1,True
        return 0,False

    def getEncodedPosition(self, position):
        encodedPosition = np.stack((position == -1, position == 0, position == 1)).astype(np.float32)
        return encodedPosition

