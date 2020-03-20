import chess
import random
from time import sleep
import numpy as np
from numba import jit, generated_jit
import sys
import matplotlib.pyplot as plt
from Machine import neural_network
from Machine import normalize
from time import time


def colour_num_gen(colour):
    if colour == 'white':
        colour_num = 1
    elif colour == 'black':
        colour_num = -1
    return colour_num

def eval(board,colour,mode):
    if board.is_checkmate() == True:
        return -10000
    if mode == 0:
        values = {'P': 1,
                  'N': 3,
                  'B': 3,
                  'R': 5,
                  'Q': 9,
                  'p': -1,
                  'n': -3,
                  'b': -3,
                  'r': -5,
                  'q': -9}
        initial_list = list(str(board))
        piece_list = []
        non_pieces = ['.',' ','\n','k','K']  # chars that aren't pieces, still in board (king on here because no countable val)
        for n in range(len(initial_list)):
            if initial_list[n] not in non_pieces:  # if the char isn't not a piece
                piece_list.append(initial_list[n])
        total = 0
        for piece in piece_list:
            total += values[piece]*colour
    else:
        total = 2*net.forwards_prop(np.reshape(imagify(board),[64*12]))+eval(board,colour,0)
    return total

def move_list_generator(moves):
    out = []
    for n in range(len(moves)):
        out.append(str(moves[n]))
    return out

#@generated_jit
def mover(moves,board,colour,rec,top,mode):
    values = []  # list of the values for all moves
    for n in range(len(moves)):
        new_board = board.copy()
        new_board.push_uci(moves[n])
        if rec == 0:
            values.append(eval(new_board,colour,mode))  # gives a value to each move
        else:
            new_moves = move_list_generator(list(new_board.legal_moves))
            if len(new_moves) > 0:
                values.append(mover(new_moves,new_board,colour*-1,rec-1,False,mode)*-1)
    if top == True:
        values = np.array([values])[0]
        moves = np.array([moves])[0]
        try:
            moves = random.choice(moves[values==max(values)])# takes only the moves that have the highest value
        except IndexError:
            moves = moves[np.argmax(values)]
    else:
        moves = max(values)
    return moves

def imagify(board):
    initial_list = list(str(board))
    square_list = np.array([])
    non_pieces = [' ','\n']  # chars that aren't pieces, still in board
    for n in range(len(initial_list)):
        if initial_list[n] not in non_pieces:  # if the char isn't not a piece
            square_list = np.append(square_list,initial_list[n])
    piece_list = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    out = [np.array([square_list==piece],dtype='int')  # for every piece, is the square on the board that piece
           for piece in piece_list]
    return out

def game(mode):
    move_count = 0
    images = np.array([])
    board = chess.Board()
    while True:
        images = np.append(images,imagify(board))
        if move_count % 2 == 0:
            move_colour = 'white'
        else:
            move_colour = 'black'
        moves = board.legal_moves
        if mode < 2:
            if len(list(moves)) != 0:
                moves = move_list_generator(list(moves))
                if mode == 0:
                    depth = 2-2*np.kron(colour_num_gen(move_colour),0)
                else:
                    depth = 2
                move = mover(moves,board,colour_num_gen(move_colour),depth,True,mode)
                board.push_uci(move)
                print(board)
                move_count += 1
        if mode > 1:
            if move_colour == 'white':
                if len(list(moves)) != 0:
                    moves = move_list_generator(list(moves))
                    move = mover(moves, board, colour_num_gen(move_colour), 2, True, mode)
                    board.push_uci(move)
                    move_count += 1
                    print(move)
                    print(board)
                else:
                    print("Draw by stalemate")
                    RESULT = 0.5
                    break
            else:
                move = str(input("move?"))
                board.push_uci(move)
                move_count += 1
                print(move)
                print(board)
        if True in [board.is_checkmate()]:
            print("Game Over,",move_colour,"wins")
            RESULT = np.kron(colour_num_gen(move_colour),2)  # if white, 1, if black, 0
            break
        elif True in [board.is_stalemate(),board.is_insufficient_material(),board.is_seventyfive_moves(),board.is_fivefold_repetition()]:
           print("draw")
           RESULT = 0.5
           break
    return images, RESULT, move_count

count = 0
net = neural_network()
net.init([64*12,256,1])
'''
for n in range(50):
    print(n)
    t = time()
    try:
        data = game(0)
    except ValueError:
        pass
    images = data[0]
    for i in range(data[2]):
        net.train(images[i*64*12:(i+1)*64*12],data[1],0.1)
    net.save('octo_weights','octo_bias')
    print(time()-t)
'''
net.load_data('octo_weights_net.npy','octo_bias_net.npy')
board = chess.Board()
print(eval(board,1,1))
for n in range(100):
    print(n)
    t = time()
    try:
        data = game(1)
    except ValueError:
        pass
    images = data[0]
    for i in range(data[2]):
        net.train(images[i*64*12:(i+1)*64*12],data[1],0.1)
    net.save('octo_weights_net','octo_bias_net')
    print(time()-t)


net.load_data('octo_weights_net.npy','octo_bias_net.npy')
game(2)