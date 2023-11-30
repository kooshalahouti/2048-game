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
        if move_number == 0:
            return 1, self.TM_SCALE_PARAM if self.mode != 'ucb' else self.UCB_TM_SCALE_PARAM

        if self.mode == 'ucb':
            search_depth = int(self.UCB_SCALER_PARAM / np.sqrt(move_number))
            total_moves = self.UCB_TM_SCALE_PARAM

        else:
            search_depth = int(self.SCALER_PARAM / np.sqrt(move_number))
            total_moves = self.TM_SCALE_PARAM

        return search_depth, total_moves

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

        move_func = np.random.choice(gf.get_moves())
        new_board, move_made, _ = move_func(np.copy(board))
        if move_made:
            return evaluation.evaluate_state(new_board) + 0.9 * MCTS.simulate_move(new_board, search_depth - 1)
        else:
            return evaluation.evaluate_state(board)
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
        ucb_scores = []
        for move in moves:
            move_score = move[0]
            move_visits = move[1]
            if move_visits == 0:
                exploration_term = np.inf
            else:
                exploration_term = self.C_CONSTANT * \
                    np.sqrt(np.log(total_visits) / move_visits)
            ucb_scores.append(move_score + exploration_term)

        return np.array(ucb_scores)

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
        moves = gf.get_moves()
        scores = np.zeros(len(moves))
        visits = np.zeros(len(moves))

        for _ in range(total_moves):
            for i, move_func in enumerate(moves):
                new_board, move_made, _ = move_func(np.copy(board))
                if move_made:
                    scores[i] += MCTS.simulate_move(new_board, search_depth)
                    visits[i] += 1

        non_zero_visits = np.where(visits != 0)
        scores[non_zero_visits] /= visits[non_zero_visits]
        return moves[np.argmax(scores)]

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
        moves = gf.get_moves()
        move_data = [(0, 0) for _ in range(len(moves))]
        total_visits = 0

        for _ in range(total_moves):
            ucb_scores = self.ucb(move_data, total_visits)
            chosen_move_index = np.argmax(ucb_scores)
            chosen_move = moves[chosen_move_index]

            new_board, move_made, _ = chosen_move(np.copy(board))
            if move_made:
                move_data[chosen_move_index] = (move_data[chosen_move_index][0] + MCTS.simulate_move(
                    new_board, search_depth), move_data[chosen_move_index][1] + 1)
            total_visits += 1

        return chosen_move
        # raise NotImplementedError("MCTS v2 not implemented yet.")
