import board
import bots
import chess
import sys
# import time


# Disable buffering
class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)


def main():
    pos = board.Board(chess.STARTING_FEN)

    engine = bots.RandoBot()

    our_time, opp_time = 1000, 1000  # time in centi-seconds

    # print name of chess engine
    print('CHESSBOT VORAPSAK')

    cmdstack = []
    while True:
        if cmdstack:
            cmd = cmdstack.pop()
        else:
            cmd = input()

        if cmd == 'quit':
            break

        elif cmd == 'uci':
            print('uciok')

        elif cmd == 'isready':
            print('readyok')

        elif cmd == 'ucinewgame':
            cmdstack.append('position fen ' + chess.STARTING_FEN)

        elif cmd.startswith('position'):
            params = cmd.split(' ', 2)
            if params[1] == 'fen':
                fen = params[2]
                pos = chess.Board(fen)

        elif cmd.startswith('go'):
            # parse parameters
            params = cmd.split(' ')
            if len(params) == 1:
                continue

            # startthink = time.time()

            move = engine.choose_move(pos)

            # endthink = time.time()

            # We only resign once we are mated.. That's never?
            pos.push(move)
            if pos.is_game_over():
                print('resign')
            else:
                print('bestmove ' + move.uci())

        elif cmd.startswith('time'):
            our_time = int(cmd.split()[1])

        elif cmd.startswith('otim'):
            opp_time = int(cmd.split()[1])

        else:
            pass


if __name__ == '__main__':
    main()

# Usage:
# In chess GUI of choice, set UCI engine command line to:
# /path/to/myvirtualenv/bin/python uci.py
