import numpy as np

import evaluation
import game_functions as gf


class Expectimax:
    def __init__(self, board):
        # You may change this parameter to scale the depth to which the agent searches.
        self.DEPTH_BASE_PARAM = 1
        # You may change this parameter to scale depth to which the agent searches.
        self.SCALER_PARAM = 400
        self.board = board

    def get_depth(self, move_number):
        """
        Returns the depth to which the agent should search for the given move number.
        ...
        :type move_number: int
        :param move_number: The current move number.
        :return: The depth to which the agent should search for the given move number.
        """
        # TODO: Complete get_depth function to return the depth to which the agent should search for the given move number.
        # Hint: You may need to use the DEPTH_BASE_PARAM constant.
        return self.DEPTH_BASE_PARAM + int(move_number / self.SCALER_PARAM)

        # raise NotImplementedError("Get depth not implemented yet.")

    def ai_move(self, board, move_number):
        depth = self.get_depth(move_number)
        score, action = self.expectimax(board, depth, 1)
        return action

    def expectimax(self, board: np.ndarray, depth: int, turn: int):
        """
        Returns the best move for the given board state and turn.
        ...
        :type turn: int
        :type depth: int
        :type board: np.ndarray
        :param board: The board state for which the best move is to be found.
        :param depth: Depth to which agent takes actions for each move
        :param turn: The turn of the agent. 1 for AI, 0 for computer.
        :return: Returns the best move and score we can obtain by taking it, for the given board state and turn.
        """

        # TODO: Complete expectimax function to return the best move and score for the given board state and turn.
        # Hint: You may need to implement minimizer_node and maximizer_node functions.
        # Hint: You may need to use the evaluation.evaluate_state function to score leaf nodes.
        # Hint: You may need to use the gf.terminal_state function to check if the game is over.

        if gf.terminal_state(board) or depth == 0:
            return evaluation.evaluate_state(board), None

        if turn == 1:
            return self.maximizer_node(board, depth)
        else:
            return self.chance_node(board, depth)
        # raise NotImplementedError("Expectimax not implemented yet.")

    def maximizer_node(self, board: np.ndarray, depth: int):
        """
        Returns the best move for the given board state and turn.
        ...
        :type depth: int
        :type board: np.ndarray
        :param board: The board state for which the best move is to be found.
        :param depth: Depth to which agent takes actions for each move
        :return: Returns the move with highest score, for the given board state.
        """

        # TODO: Complete maximizer_node function to return the move with highest score, for the given board state.
        # Hint: You may need to use the gf.get_moves function to get all possible moves.
        # Hint: You may need to use the gf.add_new_tile function to add a new tile to the board.
        # Hint: You may need to use the np.copy function to create a copy of the board.
        # Hint: You may need to use the np.inf constant to represent infinity.
        # Hint: You may need to use the max function to get the maximum value in a list.

        best_score = -np.inf
        # best_action = None

        for action in gf.get_moves():
            new_board = np.copy(board)
            move_board, score = gf.move(new_board, action)
            # move_board = gf.move(new_board, action)

            if move_board is not None:
                _, child_action = self.expectimax(move_board, depth - 1, 0)
                if child_action is not None and score + child_action > best_score:
                    best_score = score + child_action
                    # best_action = action
        return best_score

        # raise NotImplementedError("Maximizer node not implemented yet.")

    def chance_node(self, board: np.ndarray, depth: int):
        """
        Returns the expected score for the given board state and turn.
        ...
        :type depth: int
        :type board: np.ndarray
        :param board: The board state for which the expected score is to be found.
        :param depth: Depth to which agent takes actions for each move
        :return: Returns the expected score for the given board state.
        """

        # TODO: Complete chance_node function to return the expected score for the given board state.
        # Hint: You may need to use the gf.get_empty_cells function to get all empty cells in the board.
        # Hint: You may need to use the gf.add_new_tile function to add a new tile to the board.
        # Hint: You may need to use the np.copy function to create a copy of the board.
        total_score = 0
        empty_cells = gf.get_empty_cells(board)
        num_empty_cells = len(empty_cells)

        for cell in empty_cells:
            new_board = np.copy(board)
            gf.add_new_tile(new_board, cell, 2)

            _, score = self.expectimax(new_board, depth - 1, 1)
            total_score += score

        return total_score / num_empty_cells
        # raise NotImplementedError("Chance node not implemented yet.")
