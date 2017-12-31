import chess
import math
import random
import numpy as np


def alphabeta_search(board, d=4, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    player = board.turn

    # The default test cuts off at depth d or at a terminal state
    def cutoff_test(board, depth):
        return depth > d or board.is_game_over()

    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        maxv = -math.inf
        for a in state.legal_moves:
            state.push(a)
            maxv = max(maxv, min_value(state, alpha, beta, depth+1))
            state.pop()
            if maxv >= beta:
                return maxv
            alpha = max(alpha, maxv)
        return maxv

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        minv = math.inf
        for a in state.legal_moves:
            state.push(a)
            minv = min(minv, max_value(state, alpha, beta, depth+1))
            state.pop()
            if minv <= alpha:
                return minv
            beta = min(beta, minv)
        return minv

    best = -math.inf
    best_move = None
    for m in board.legal_moves:
        board.push(m)
        v = min_value(board, -math.inf, math.inf, 0)
        board.pop()
        if v > best:
            best = v
            best_move = m
    return best_move


# plays random, but always prefers a capture move
class CaptureBot(object):
    def __init__(self):
        pass

    # return a number that goes up and down
    def eval_board(self, board):
        # add up value of black pcs vs white pcs
        score = 0.0
        score += len(board.pieces(chess.PAWN, chess.WHITE))*1.0
        score -= len(board.pieces(chess.PAWN, chess.BLACK))*1.0
        score += len(board.pieces(chess.KNIGHT, chess.WHITE))*3.2
        score -= len(board.pieces(chess.KNIGHT, chess.BLACK))*3.2
        score += len(board.pieces(chess.BISHOP, chess.WHITE))*3.0
        score -= len(board.pieces(chess.BISHOP, chess.BLACK))*3.0
        score += len(board.pieces(chess.ROOK, chess.WHITE))*5.0
        score -= len(board.pieces(chess.ROOK, chess.BLACK))*5.0
        score += len(board.pieces(chess.QUEEN, chess.WHITE))*8.0
        score -= len(board.pieces(chess.QUEEN, chess.BLACK))*8.0

        score = score + random.uniform(-0.1, 0.1)
        if board.turn == chess.WHITE:
            score = score * -1
        return score

    def choose_move(self, board):
        # do a tree search and just go for max num of captures.
        # only search 3 levels deep.
        self.max_depth = 3

        return alphabeta_search(board, self.max_depth, lambda b: self.eval_board(b))
