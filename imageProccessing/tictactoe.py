#!/usr/bin/env python

# Author: Team3
# Date: 2024-03-08

#Random Empty File. 
#! /usr/bin/env python


#note: this is a modified version of ChatGPT code

import random

def check_winner(board, player):
    # Check rows
    for row in board:
        if all(cell == player for cell in row):
            return True

    # Check columns
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True

    # Check diagonals
    if all(board[i][i] == player for i in range(3)) or all(board[i][2-i] == player for i in range(3)):
        return True

    return False

def check_draw(board):
    for row in board:
        for cell in row:
            if cell == ' ':
                return False
    return True

def computer_move(board, computer):
    # Check if computer can win in the next move
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = computer
                if check_winner(board, computer):
                    return (i, j)
                board[i][j] = ' '

    # Check if player can win in the next move
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'X'  # Assume player's move
                if check_winner(board, 'X'):
                    board[i][j] = computer
                    return (i, j)
                board[i][j] = ' '

    # Try to take corners
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    random.shuffle(corners)
    for corner in corners:
        if board[corner[0]][corner[1]] == ' ':
            return corner

    # Try to take center
    if board[1][1] == ' ':
        return (1, 1)

    # Take any available edge
    edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
    random.shuffle(edges)
    for edge in edges:
        if board[edge[0]][edge[1]] == ' ':
            return edge

def play(board, player):
	#board = [[' ']*3 for _ in range(3)]
     
	#set computer variable oposite to player
	computer = 'O' if player == 'X' else 'X'

	#check if player has won
	if check_winner(board, player):
		return [0,0], 1
     
	#program to determine which space robot should place 
	row, col = computer_move(board, computer)
	space_to_play = [row, col]
     
	board[row][col] = computer
    
	#check if computer has won
	if check_winner(board, player):
		return space_to_play, 2

	#check if draw
	if check_draw(board):
		return space_to_play, 3
    
	#no winner yet, continue play
	return space_to_play, 0

