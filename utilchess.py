import chess


class UtilChess:

    @staticmethod
    def get_pieces_positions_value(board):
        pieces_positions = []
        pieces_positions.append(UtilChess.get_piece_positions_value(
            board, chess.PAWN, chess.WHITE))
        pieces_positions.append(UtilChess.get_piece_positions_value(
            board, chess.PAWN, chess.BLACK))
        pieces_positions.append(UtilChess.get_piece_positions_value(
            board, chess.ROOK, chess.WHITE))
        pieces_positions.append(UtilChess.get_piece_positions_value(
            board, chess.ROOK, chess.BLACK))
        pieces_positions.append(
            UtilChess.get_piece_positions_value(board, chess.KNIGHT, chess.WHITE))
        pieces_positions.append(
            UtilChess.get_piece_positions_value(board, chess.KNIGHT, chess.BLACK))
        pieces_positions.append(
            UtilChess.get_piece_positions_value(board, chess.BISHOP, chess.WHITE))
        pieces_positions.append(
            UtilChess.get_piece_positions_value(board, chess.BISHOP, chess.BLACK))
        pieces_positions.append(
            UtilChess.get_piece_positions_value(board, chess.QUEEN, chess.WHITE))
        pieces_positions.append(
            UtilChess.get_piece_positions_value(board, chess.QUEEN, chess.BLACK))
        pieces_positions.append(UtilChess.get_piece_positions_value(
            board, chess.KING, chess.WHITE))
        pieces_positions.append(
            UtilChess.get_piece_positions_value(board, chess.KING, chess.BLACK))

        return pieces_positions

    @staticmethod
    def get_piece_positions_value(board, piece, color):
        return int(board.pieces(piece, color))

    @staticmethod
    def get_chess_material_positions(game, pgn):
        nb_games = 0
        nb_moves = 0
        board = chess.Board()
        all_pieces_positions = []
        all_result = []
        while game:
            if game.variations:
                game_model = game
                nb_games += 1
                # Get the first move (initialisation)
                move1 = game.variations[0].move
                # print("Learning moves from %s game " % game_model.headers["FICSGamesDBGameNo"])
                while move1 is not None:
                    board.push(move1)
                    # print("Current move %r" % move1)
                    pieces_positions = UtilChess.get_pieces_positions_value(board)
                    # print("Pieces positions %r" % pieces_positions)
                    # Here we retrieve the result and we have the corresponding the chance of wininng
                    # e.g [1, 0, 0] meaning that the Black won the game
                    # e.g [0, 0, 1] meaning that the White won the game
                    result_str = game_model.headers["Result"]
                    if result_str == "0-1":
                        result = [1, 0, 0]
                    elif result_str == "1-0":
                        result = [0, 0, 1]
                    else:
                        result = [0, 1, 0]
                    all_pieces_positions.append(pieces_positions)
                    all_result.append(result)

                    # print("Result of the game %r" % result)
                    nb_moves += 1

                    # Move to the wanted position and print the next move
                    game = game.variation(move1)
                    if game.is_end():
                        move1 = None
                    else:
                        move1 = game.variations[0].move

            board = chess.Board()
            game = chess.pgn.read_game(pgn)

        return [all_pieces_positions, all_result, nb_games, nb_moves]
