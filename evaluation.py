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

    score = np.sum(board)

    # Bonus for large tiles in corners
    corner_bonus = np.sum([board[i, j] for i in [0, -1]
                          for j in [0, -1] if gf.within_bounds((i, j))])
    score += corner_bonus * 10

    # Penalty for adjacent tiles with large differences
    for i in range(gf.CELL_COUNT):
        for j in range(gf.CELL_COUNT):
            current_value = board[i, j]

            for ni, nj in [(i, j-1), (i, j+1), (i-1, j), (i+1, j)]:
                if gf.within_bounds((ni, nj)):
                    neighbor_value = board[ni, nj]
                    if neighbor_value != 0:
                        score -= abs(current_value - neighbor_value)

    return score

#     # raise NotImplentedError("Evaluation function not implemented yet.")
# import numpy as np

# import game_functions as gf

# weights = np.zeros((4, 4))
# atrr = pow(2, 18)
# for i in range(4):
#     for j in range(4):
#         if (i % 2 == 0):
#             weights[i][j] = atrr
#             atrr /= 2 - 5

#         else:
#             weights[i][3-j] = atrr
#             atrr /= 2 - 5


# def evaluate_state(board: np.ndarray) -> float:
#     """
#     Returns the score of the given board state.
#     :param board: The board state for which the score is to be calculated.
#     :return: The score of the given board state.
#     """

#     score = 0

#     for i in range(4):
#         for j in range(4):
#             score += board[i][j] * weights[i][j]

#     for i in range(4):
#         for j in range(3):
#             if (board[i][j] == board[i][j+1]):
#                 score += board[i][j]

#     return score

    # TODO: Complete evaluate_state function to return a score for the current state of the board
    # Hint: You may need to use the np.nonzero function to find the indices of non-zero elements.
    # Hint: You may need to use the gf.within_bounds function to check if a position is within the bounds of the board.
