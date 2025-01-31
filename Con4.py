import numpy as np

class Connect4:

    def __init__(self,numRows,numColumns):
        self.numRows=numRows
        self.numColumns=numColumns
        self.numMoves=numColumns

    def getLegalMoves(self,position):
        return (position[0]==0).astype(int)

    def getNextPosition(self, position, move, player):
        row=np.max(np.where(position[:, move] == 0))
        position[row,move]=player
        return position

    def countPiecesInGivenDirection(self,position,move,rowDirection,columnDirection):
    #position is the position after the move has been made. counts successive of the moving player starting at
    #the position of the new piece
        column=move
        row=np.min(np.where(position[:, move] != 0))
        player=position[row,column]
        for i in range(1,4):
            nextRow=row+i*rowDirection
            nextColumn = column + i * columnDirection
            if(
                nextRow<0 or nextRow==self.numRows or
                nextColumn < 0 or nextColumn == self.numColumns or
                position[nextRow,nextColumn]!=player
            ):
                return i-1
        return 3

    def checkIfWin(self,position,move):
        #position is the position after the move has been made
        if move == None:
            return False
        column = move
        row = np.min(np.where(position[:, move] != 0))
        return (
            ##check all possible directions
            self.countPiecesInGivenDirection(position,move,1,0)>=3 or #vertical direction
            self.countPiecesInGivenDirection(position,move, 0, 1)+
            self.countPiecesInGivenDirection(position,move, 0, -1)>= 3 or #horizontal direction
            self.countPiecesInGivenDirection(position, move, 1, 1) +
            self.countPiecesInGivenDirection(position, move, -1, -1) >= 3 or #the diagonal
            self.countPiecesInGivenDirection(position, move, -1, 1) +
            self.countPiecesInGivenDirection(position, move, 1, -1) >= 3 #the anti-diagonal
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


con4=Connect4(6,7)
position=np.zeros((con4.numRows, con4.numColumns))
position=con4.getNextPosition(position,3,1)
position=con4.getNextPosition(position,3,1)
position=con4.getNextPosition(position,3,1)
position=con4.getNextPosition(position,3,1)
