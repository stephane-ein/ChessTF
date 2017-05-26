import os
import io
import chess
import chess.pgn
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from utilchess import UtilChess


def main():
    """ main function """
    mypath = os.getcwd() + "/training/"
    board = chess.Board()
    # Get pieces (SquareByte (en gros un byte))
    squaret = board.pieces(chess.PAWN, chess.WHITE)
    # Get piece positoin (ou via list(squreset))
    int(squaret)

    # print(board)
    # print(board.legal_moves)
    # for move in board.legal_moves:
    #     print(move)

    # Init the modal data
    # The 12 rows corresponds to the several chess material
    # (pawn/queen/king/rook/bishop/kight black and white)
    x = tf.placeholder(tf.float32, [None, 12])
    # W = tf.Variable(tf.zeros([12, 3]))
    # b = tf.Variable(tf.zeros([3]))

    # Output computed (here the AI computes who think is the winner)
    # y = tf.nn.softmax(tf.matmul(x, W) + b)
    # y = tf.matmul(x, W) + b

    # Neural network with 5 layer
    K = 200
    L = 100
    M = 60
    N = 30

    W1 = tf.Variable(tf.zeros([12, K]))
    B1 = tf.Variable(tf.zeros([K]))
    W2 = tf.Variable(tf.zeros([K, L]))
    B2 = tf.Variable(tf.zeros([L]))
    W3 = tf.Variable(tf.zeros([L, M]))
    B3 = tf.Variable(tf.zeros([M]))
    W4 = tf.Variable(tf.zeros([M, N]))
    B4 = tf.Variable(tf.zeros([N]))
    W5 = tf.Variable(tf.zeros([N, 3]))
    B5 = tf.Variable(tf.zeros([3]))

    Y1 = tf.nn.relu(tf.matmul(x, W1) + B1)
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
    Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
    Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
    y = tf.matmul(Y4, W5) + B5

    # True ouput of the data modal (here the winner)
    y_ = tf.placeholder(tf.float32, [None, 3])

    # Define the cross entropy in order to compute how much we are far from the
    # real output
    # cross_entropy = tf.reduce_mean(- tf.reduce_sum(y_ *
    #                                                tf.log(y), reduction_indices=[1]))
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0))))
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y + 1e-10)))
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(
        0.003).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Stats about the game
    nb_total_game = 0
    nb_total_moves = 0
    # Get pgn files
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    nb_files = len(onlyfiles)
    files_process = 0
    for pgn_file in onlyfiles:
        print("Retrieving game from %s..." % pgn_file)
        try:
            pgn = open(mypath + pgn_file)
            game = chess.pgn.read_game(pgn)

            # Retrieve all the game moves from teh training data set
            all_pieces_positions, all_result, nb_games, nb_moves = UtilChess.get_chess_material_positions(
                game, pgn)
            nb_total_game += nb_games
            nb_total_moves += nb_moves

            # Train the AI
            sess.run(train_step, feed_dict={
                x: all_pieces_positions, y_: all_result})

            # Print the wieght generated and the bias
            # print("Weight for %s game %r" % (nb_games, sess.run(W)))
            # print("Bias for %s game %r" % (nb_games, sess.run(b)))
            print("%r games for file %s" % (nb_games, pgn_file))
        except Exception:
            print("[ERROR] Error when parsing file: %r" % pgn_file)
        files_process += 1
        print("%s/%s files process" % (files_process, nb_files))

    print("%r game learned" % nb_total_game)
    print("%r moves learned" % nb_total_moves)
    # print("Final weight after %s game %r" % (nb_total_game, sess.run(W)))
    # print("FInal bias after %s game %r" % (nb_total_game, sess.run(b)))

    # Now let's test the AI and 0 it's a Black win, 1, draw and 2 white
    # win.
    # pgn = open(mypath + onlyfiles[0])
    # mypathtest = os.getcwd() + "/test/"
    # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # pgn = open(mypathtest + onlyfiles[0])
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # test_all_pieces_position = []
    # test_all_result = []
    # game = chess.pgn.read_game(pgn)
    # nb_valid_predictions = 0
    # nb_no_valid_prediction = 0
    # nb_predictions = 0
    # while game.variation:
    #     # while False:
    #     # while nb_predictions == 0:
    #     move1 = game.variations[0].move
    #     game_model = game
    #     while move1 is not None:
    #         board = chess.Board()
    #         board.push(move1)
    #         pieces_positions = UtilChess.get_pieces_positions_value(board)
    #         # pieces_positions = [0, 0, 0, 0, 0, 0, 0, 0, 8, 576460752307617792, 0, 0]
    #         prediction = tf.argmax(y, 1)
    #         prediction_result = prediction.eval(
    #             feed_dict={x: [pieces_positions]})
    #         prediction_str = game_model.headers["Result"]
    #         print("Result %s and piece_positions %r" %
    #               (prediction_result, pieces_positions))
    #         print("Result %r for %r" %
    #               (prediction_str, game_model.headers["FICSGamesDBGameNo"]))

    #         if prediction_result == 2 and prediction_str == "1-0":
    #             nb_valid_predictions += 1
    #         elif prediction_result == 1 and prediction_str == "1/2-1/2":
    #             nb_valid_predictions += 1
    #         elif prediction_result == 0 and prediction_str == "0-1":
    #             nb_valid_predictions += 1
    #         else:
    #             nb_no_valid_prediction += 1
    #         nb_predictions += 1
    #         game = game.variation(move1)
    #         if game.is_end():
    #             move1 = None
    #         else:
    #             move1 = game.variations[0].move
    #     game = chess.pgn.read_game(pgn)
    # Print the AI verification
    # print("So we have %s prediction and %s valid prediction and %s not valid prediction" % (
    #     nb_predictions, nb_valid_predictions, nb_no_valid_prediction))
    # print("Which it's %s accurate" %
    #       ((nb_valid_predictions * 100) / nb_predictions))

    mypathtest = os.getcwd() + "/test/"
    onlyfiles = [f for f in listdir(mypathtest) if isfile(join(mypathtest, f))]
    pgn = open(mypathtest + onlyfiles[0])
    # Retrieve all the game moves from the training data set and computes the
    # accurary
    all_pieces_positions, all_result, nb_games, nb_moves = UtilChess.get_chess_material_positions(
        game, pgn)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy %r" % sess.run(accuracy, feed_dict={x: all_pieces_positions,
                                                        y_: all_result}))
    retrieve_winner_from_position(
        y, x, [12582912, 140737488355328, 0, 34359738368, 2097152, 0, 0, 0, 0, 0, 8192, 536870912], 2)
    retrieve_winner_from_position(
        y, x, [0, 0, 2199023255552, 268435456, 0, 0, 0, 0, 0, 0, 8589934592, 34359738368], 1)
    retrieve_winner_from_position(y, x, [2688679936, 27725293795934208, 8193,
                                         1152921504875282432, 0, 0, 0, 0, 512, 4398046511104, 16384, 4611686018427387904], 0)
    retrieve_winner_from_position(y, x, [57600, 36066184709275648, 32, 1729382256910270464,
                                         137438953472, 268435456, 16, 0, 0, 0, 64, 9223372036854775808], 0)
    retrieve_winner_from_position(y, x, [962072674304, 36066180414308352,
                                         0, 262144, 0, 0, 0, 0, 0, 16777216, 536870912, 4503599627370496], 0)


def retrieve_winner_from_position(y, x, pieces_positions, result):
    # Manuel check
    prediction = tf.argmax(y, 1)
    print("Prediction is %s and the result is %s" %
          (prediction.eval(feed_dict={x: [pieces_positions]}), result))


if __name__ == "__main__":
    main()
