import numpy as np

import evaluation
import game_functions as gf


class MCTS:
    def __init__(self, board, mode='ucb'):
        # You may change this parameter to scale the exploration term in the UCB formula.
        self.C_CONSTANT = 2
        # You may change this parameter to scale the depth to which the agent searches.
        self.SD_SCALE_PARAM = 5
        # You may change this parameter to scale the depth to which the agent searches.
        self.TM_SCALE_PARAM = 5
        # You may change this parameter to scale the depth to which the agent searches.
        self.SCALER_PARAM = 200
        # You may change this parameter to scale the depth to which the agent searches.
        self.UCB_SD_SCALE_PARAM = 5
        # You may change this parameter to scale the depth to which the agent searches.
        self.UCB_TM_SCALE_PARAM = 5
        # You may change this parameter to scale the depth to which the agent searches.
        self.UCB_SCALER_PARAM = 300
        self.board = board
        self.mode = mode

    def get_search_params(self, move_number: int) -> (int, int):
        """
        Returns the depth to which the agent should search for the given move number.
        ...
        :type move_number: int
        :param move_number: The current move number.
        :return: The depth to which the agent should search for the given move number.
        """
        # TODO: Complete get_search_params function to return the depth to which the agent should search for the given move number.
        # Hint: You may want to use the self.SD_SCALE_PARAM, self.SL_SCALE_PARAM, and self.SCALER_PARAM parameters.
        # Hint: You may want to use the self.UCB_SPM_SCALE_PARAM, self.UCB_SL_SCALE_PARAM, and self.UCB_SCALER_PARAM parameters.
        # Hint: You may want to use the self.mode parameter to check which mode the agent is on.
        if self.mode == 'ucb':
            return self.UCB_SD_SCALE_PARAM, self.UCB_TM_SCALE_PARAM * int(move_number / self.UCB_SCALER_PARAM)

        else:
            return self.SD_SCALE_PARAM, self.TM_SCALE_PARAM * int(move_number / self.SCALER_PARAM)

        # raise NotImplementedError("Get search params not implemented yet.")

    def ai_move(self, board, move_number):
        search_depth, total_moves = self.get_search_params(move_number)
        if self.mode == 'ucb':
            best_move = self.mcts_v2(board, total_moves * 4, search_depth)
        else:
            best_move = self.mcts_v0(board, total_moves, search_depth)
        return best_move

    @staticmethod
    def simulate_move(board: np.ndarray, search_depth: int) -> float:
        """
        Returns the score of the given board state.
        :param board: The board state for which the score is to be calculated.
        :param search_depth: The depth to which the agent should search for the given board state.
        :return: The score of the given board state.
        """
        # TODO: Complete simulate_move function to simulate a move and return the score of the given board state.
        # Hint: You may want to use the gf.random_move function to simulate a random move.
        # Hint: You may want to use the evaluation.evaluate_state function to score a board.
        # Hint: You may want to use the move_made returned from the gf.random_move function to check if a move was made.
        # Hint: You may want to use the gf.add_new_tile function to add a new tile to the board.

        if search_depth == 0 or gf.terminal_state(board):
            return evaluation.evaluate_state(board)

        move_made, move_board = gf.random_move(board)

        if not move_made:
            return evaluation.evaluate_state(board)

        gf.add_new_tile(move_board)
        return MCTS.simulate_move(move_board, search_depth - 1)
        # raise NotImplementedError("Simulate move not implemented yet.")

    def ucb(self, moves: list, total_visits: int) -> np.ndarray:
        """
        Returns the UCB scores for the given moves.
        :param moves: The moves for which the UCB scores are to be calculated.
        :param total_visits: The total number of visits for all moves.
        :return: The UCB scores for the given moves.
        """
        # TODO: Complete ucb function to return the UCB scores for the given moves.
        # Hint: You may want to use the self.C_CONSTANT parameter to scale the exploration term in the UCB formula.
        # Hint: You may want to use np.inf to represent infinity.
        # Hint: You may want to use np.sqrt to calculate the square root of a number.
        # Hint: You may want to use np.log to calculate the natural logarithm of a number.
        ucb_scores = np.zeros(len(moves))
        for i, move in enumerate(moves):
            exploration_term = self.C_CONSTANT * \
                np.sqrt(np.log(total_visits) / move['visits'])
            ucb_scores[i] = move['average_score'] + exploration_term
        return ucb_scores
        # raise NotImplementedError("UCB not implemented yet.")

    def mcts_v0(self, board: np.ndarray, total_moves: int, search_depth: int):
        """
        Returns the best move for the given board state.
        ...
        :type search_depth: int
        :type total_moves: int
        :type board: np.ndarray
        :param board: The board state for which the best move is to be found.
        :param total_moves: The total number of moves to be simulated.
        :param search_depth: The depth to which the agent should search for the given board state.
        :return: Returns the best move for the given board state.
        """
        # TODO: Complete mcts_v0 function to return the best move for the given board state.
        # Hint: You may want to use the gf.get_moves function to get all possible moves.
        # Hint: You may want to use the gf.add_new_tile function to add a new tile to the board.
        # Hint: You may want to use the self.simulate_move function to simulate a move.
        # Hint: You may want to use the np.argmax function to get the index of the maximum value in an array.
        # Hint: You may want to use the np.zeros function to create an array of zeros.
        # Hint: You may want to use the np.copy function to create a copy of a numpy array.
        moves_info = [{'move': move, 'visits': 0, 'total_score': 0,
                       'average_score': 0} for move in gf.get_moves()]
        for _ in range(total_moves):
            moved_board = np.copy(board)
            move_made, score = gf.move(moved_board, moves_info['move'])
            if move_made:
                gf.add_new_tile(moved_board)
                simulation_score = self.simulate_move(
                    moved_board, search_depth)
                moves_info['visits'] += 1
                moves_info['total_score'] += simulation_score

        ucb_scores = self.ucb(moves_info, total_moves)
        best_move_index = np.argmax(ucb_scores)
        return moves_info[best_move_index]['move']

        # raise NotImplementedError("MCTS v0 not implemented yet.")

    def mcts_v2(self, board, total_moves, search_depth):
        """
        Returns the best move for the given board state.
        ...
        :type search_depth: int
        :type total_moves: int
        :type board: np.ndarray
        :param board: The board state for which the best move is to be found.
        :param total_moves: The total number of moves to be simulated.
        :param search_depth: The depth to which the agent should search for the given board state.
        :return: Returns the best move for the given board state.
        """
        # TODO: Complete mcts_v2 function to return the best move for the given board state.
        # Hint: You may want to use the gf.get_moves function to get all possible moves.
        # Hint: You may want to use the gf.add_new_tile function to add a new tile to the board.
        # Hint: You may want to use the self.simulate_move function to simulate a move.
        # Hint: You may want to use the np.argmax function to get the index of the maximum value in an array.
        # Hint: You may want to use the np.copy function to create a copy of a numpy array.
        # Hint: You may want to use the self.ucb function to get the UCB scores for the given moves.
        moves_info = [{'move': move, 'visits': 0, 'total_score': 0,
                       'average_score': 0} for move in gf.get_moves()]
        for _ in range(total_moves):
            for move_info in moves_info:
                moved_board = np.copy(board)
                move_made, score = gf.move(moved_board, move_info['move'])
                if move_made:
                    gf.add_new_tile(moved_board)
                    simulation_score = self.simulate_move(
                        moved_board, search_depth)
                    move_info['visits'] += 1
                    move_info['total_score'] += simulation_score
        ucb_scores = self.ucb(move_info, total_moves)
        best_move_index = np.argmax(ucb_scores)
        return moves_info[best_move_index]['move']
        # raise NotImplementedError("MCTS v2 not implemented yet.")
