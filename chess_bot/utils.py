def play_game(board, player1, player2, verbose=False):
    players = [player1, player2]
    
    turn = 0
    while not board.is_game_over():
        if verbose: print(turn)
        
        moving_player = players[board.turn]
        move = moving_player.choose_move(board)
        board.push(move)
        
        turn +=1
        
    return board, turn