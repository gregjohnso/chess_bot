from scipy.special import entr
import numpy as np


def pick_rando_move(board):
    legal_moves = list(board.legal_moves)
    move_ind = np.random.choice(len(legal_moves))
    
    return legal_moves[move_ind]

class RandoBot(object):
    def __init__(self):
        pass
    
    def choose_move(self, board):
        return pick_rando_move(board)

class SKLearnBot(object):
    def __init__(self, model, state='explore'):
        self.model = model
        self.state = 'explore'
        
    def choose_move(self, board):
        next_moves_list = board.vec_next_moves()
        
        move_scores = np.zeros(len(next_moves_list))
        
        moves_flat = [move[0].flatten() for move in next_moves_list]
        moves_flat = np.vstack(moves_flat)
        
        scores = self.model.predict_proba(moves_flat)
        
        if self.state == 'explore':
            entropy = entr(scores).sum(axis=1)/np.log(2)
            ind = np.argmax(entropy)
        elif self.state == 'exploit':
            ind = np.argmax(scores[:,1])
        elif self.state == 'wexploit': #weighted exploitation
            scores = scores[:,1]
            scores = scores/sum(scores)
            ind = np.random.choice(np.arange(0, len(scores)), p = scores)
        
        legal_moves = list(board.legal_moves)
        
        return legal_moves[ind]
                                       
    def train_from_board(self, boards):
#       need to finish
        
        all_moves = list()
        all_won_game = list()
        
        for board in boards:
            #do this so we dont screw up the board objects outside of this
            board = board.copy()
            
            winner = board.who_won()
            moves = board.vec_history()
            
            moves_flat = [move[0].flatten() for move in moves]
            moves_flat = np.vstack(moves_flat)
            
            whose_turn = np.array([move[1] for move in moves])
            
            winning_moves = whose_turn == winner
            
            all_moves.append(moves_flat)
            all_won_game.append(winning_moves)
            
        all_moves = np.vstack(all_moves)
        all_won_game = np.hstack(all_won_game)
        
        self.model.fit(all_moves, all_won_game)
        