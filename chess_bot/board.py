from chess import Board as PyChessBoard
import numpy as np

import pdb

class Board(PyChessBoard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
       
    def __len__(self):
        return self.fullmove_number*2 - self.turn

    def vec(self):
        return board2vec(self)
    
    def vec_next_moves(self):
        return board2vec_next_moves(self)
        
    def vec_history(self):
        return board2vec_history(self)
    
    def who_won(self):
        return who_won(self)

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
    
def piece2vec(piece_str):
    """Converts a string representation of a piece to a one-hot representation"""
    
    piece_vec = np.zeros(len(piece_dictionary))
    
    if piece_str in piece_dictionary:
        piece_vec[piece_dictionary[piece_str]] = True
    
    return piece_vec


def board2vec(board):
    """Converts a board object to a 12 pieces x rows x columns one-hot representation"""
    board_state = str(board)

    # print(board_state)
    board_state = board_state.split('\n')
    board_state = np.array([row.split(' ') for row in board_state])

    has_piece = np.array(board_state) != '.'
    i, j = np.where(has_piece)
    
    pieces = [piece_dictionary[piece] for piece in board_state[has_piece]]
    
    board_vector = np.zeros([12, 8, 8])
    board_vector[pieces, i, j] = 1
    
    return [board_vector, board.turn]
            
def vec2board(board, turn):    
    """Converts a 12 pieces x rows x columns one-hot representation to a board object"""
#     todo, not necessary
    pass


def board2vec_next_moves(board):
    """Returns a list of next legal moves as board vectors"""
    
    move_list = list()
    turn_list = list()
    
    for move in board.legal_moves:
        board_tmp = board.copy()
        board_tmp.push(move)
        
        board_vec, turn = board2vec(board_tmp)
        move_list.append(board_vec)
        turn_list.append(turn)
        
    return move_list, turn_list

def board2vec_history(board):
    board = board.copy()
    
    board_len = len(board)
    
    move_list = list()
    turn_list = list()
    isempty = False
    c = 0
    
    #pop off every move, and get it's vector form
    for i in range(0, board_len):
        
        board_vec, turn = board2vec(board)
        
        move_list.append(board_vec)
        turn_list.append(turn)
        
        if i < (board_len-1):
            board.pop()
    
    move_list = move_list[::-1]
    turn_list = turn_list[::-1]
            
    return move_list, turn_list
    
    
def who_won(board):
    """Returns who won 0: black, 1: white, 2: tie, """
    win_dict = {'0-1':0, '1-0':1, '1/2-1/2':2}
    
    return win_dict[board.result()]    