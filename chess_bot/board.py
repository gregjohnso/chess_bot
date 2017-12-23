from chess import Board as PyChessBoard
import numpy as np


class Board(PyChessBoard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def vec(self):
        return board2vec(self)
    
    def vec_next_moves(self):
        return board2vec_next_moves(self)
        
    def vec_history(self):
        return board2vec_history(self)
    
    def who_won(self):
        return who_won(self)


def piece2vec(piece_str):
    """Converts a string representation of a piece to a one-hot representation"""
    
    piece_dictionary = { #ignore empty position
                        'B': 0, 
                        'K': 1,
                        'N': 2,
                        'P': 3,
                        'Q': 4,
                        'R': 5,
                        'b': 6,
                        'k': 7,
                        'n': 8,
                        'p': 9,
                        'q': 10,
                        'r': 11}
    
    piece_vec = np.zeros(len(piece_dictionary)) > 0
    
    if piece_str in piece_dictionary:
        piece_vec[piece_dictionary[piece_str]] = True
    
    return piece_vec


def board2vec(board):
    """Converts a board object to a 12 pieces x rows x columns one-hot representation"""
    board_state = str(board)

    # print(board_state)
    board_state = board_state.split('\n')
    board_state = [row.split(' ') for row in board_state]

    board_vector = np.zeros([12, 8, 8])

    # for every row
    for i in range(0, len(board_state)):
        #for every column
        for j in range(0, len(board_state[0])):
            piece_str = board_state[i][j]

            board_vector[piece2vec(piece_str), i,j] = 1

    return [board_vector, board.turn]
            
def vec2board(board, turn):    
    """Converts a 12 pieces x rows x columns one-hot representation to a board object"""
#     todo, not necessary
    pass


def board2vec_next_moves(board):
    """Returns a list of next legal moves as board vectors"""
    
    next_moves_vec = list()
    
    for move in board.legal_moves:
        board_tmp = board.copy()
        board_tmp.push(move)
        
        next_move_vec = board2vec(board_tmp)
        next_moves_vec.append(next_move_vec)
        
    return next_moves_vec

def board2vec_history(board):
    board = board.copy()
    
    move_list = list()
    
    isempty = False
    c = 0
    while not isempty:
        c+=1
        move_list.append(board2vec(board))
        try:
            board.pop()
        except:
            isempty = True
            
    return move_list[::-1]
    
    
def who_won(board):
    """Returns who won 0: black, 1: white, 2: tie, """
    win_dict = {'0-1':0, '1-0':1, '1/2-1/2':2}
    
    return win_dict[board.result()]    