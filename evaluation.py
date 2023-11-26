import numpy as np

import game_functions as gf


def evaluate_state(board: np.ndarray) -> float:
    """
    Returns the score of the given board state.
    :param board: The board state for which the score is to be calculated.
    :return: The score of the given board state.
    """
    # TODO: Complete evaluate_state function to return a score for the current state of the board
    # Hint: You may need to use the np.nonzero function to find the indices of non-zero elements.
    # Hint: You may need to use the gf.within_bounds function to check if a position is within the bounds of the board.
    empty_cells_weight = 100
    monotonicity_weight = 100
    max_tile_weight = 1

    non_zero_indices = np.nonzero(board)
    empty_cells_score = empty_cells_weight * len(non_zero_indices[0])

    monotonicity_score = 0
    for row in board:
        for i in range(len(row) - 1):
            if gf.within_bounds((i, 0)) and gf.within_bounds((i + 1, 0)):
                monotonicity_score += np.abs(row[i] - row[i + 1])

    for col_index in range(board.shape[1]):
        col = board[:, col_index]
        for i in range(len(col) - 1):
            if gf.within_bounds((0, i)) and gf.within_bounds((0, i + 1)):
                monotonicity_score += np.abs(col[i] - col[i + 1])

    monotonicity_score *= monotonicity_weight

    max_tile_score = max_tile_weight * np.max(board)
    total_score = empty_cells_score + monotonicity_score + max_tile_score

    return total_score

    # raise NotImplementedError("Evaluation function not implemented yet.")
