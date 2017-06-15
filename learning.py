import os
import io
import chess
import chess.pgn
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from utilchess import UtilChess


class Learning:

    def __init__(self):
        # Init a board
        self.board = chess.Board()
        self.iteration = 0

    def learn(self):
        # Okay, now new have to iterate oer all the chess availabe chess
        self.explore_move()
    
    def explore_move(self):
        for move in self.board.legal_moves:
            self.board.push(move)
            self.iteration += 1
            print(self.board)
            print(self.iteration)
            print(self.board.is_game_over())
            if self.board.is_game_over():
                print(self.board.result())
                self.board.pop()
            else:
                self.explore_move()
                self.board.pop()

