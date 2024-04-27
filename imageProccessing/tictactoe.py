import random
#from imageProccessing.Player import BoardSquare

def check_winner(board, player):
    # Check rows
    rows = [
        [0, 1, 2],  # Top row
        [5, 4, 3],  # Middle row reversed
        [6, 7, 8]   # Bottom row
    ]
    for row in rows:
        if all(board[idx].tile == player for idx in row):
            return True

    # Check columns
    columns = [
        [0, 5, 6],  # First column
        [1, 4, 7],  # Second column
        [2, 3, 8]   # Third column
    ]
    for col in columns:
        if all(board[idx].tile == player for idx in col):
            return True

    # Check diagonals
    if all(board[idx].tile == player for idx in [0, 4, 8]) or all(board[idx].tile == player for idx in [2, 4, 6]):
        return True

    return False

def check_draw(board):
    return all(square.tile is not None for square in board)

def computer_move(board, computer):
    # Try all positions to find a winning move or block opponent
    for idx in range(9):
        if board[idx].tile is None:
            board[idx].tile = computer
            if check_winner(board, computer):
                return idx  # Returning approximate grid position
            board[idx].tile = None

    # Prioritize corners, center, then edges
    priority = [0, 2, 6, 8, 4, 1, 3, 5, 7]  # Priority positions
    for idx in priority:
        if board[idx].tile is None:
            return idx  # Returning approximate grid position

def play(board, player):
    # Set computer variable opposite to player
    computer = 'O' if player == 'X' else 'X'

    # Check if player has won
    if check_winner(board, player):
        return [0, 0], 1  # Game over, player won

    # Determine move for computer
    index = computer_move(board, computer)
    return index
    space_to_play = [row, col]

    #board[row * 3 + col].tile = computer  # Update board with computer's move - will be updated on next run

    # Check if computer has won
    if check_winner(board, computer):
        return space_to_play, 2  # Game over, computer won

    # Check if draw
    if check_draw(board):
        return space_to_play, 3  # Game over, draw

    # No winner yet, continue play
<<<<<<< HEAD
    return space_to_play, 
=======
    return space_to_play
>>>>>>> d387e8156ebfdca6f87a3f3161c46633f48d091d
