import numpy as np

POSSIBLE_MOVES_COUNT = 4
CELL_COUNT = 4
NUMBER_OF_SQUARES = CELL_COUNT * CELL_COUNT
NEW_TILE_DISTRIBUTION = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 4])
NEW_TILE_TWO_PROBABILITY = 0.9
NEW_TILE_FOUR_PROBABILITY = 0.1


def initialize_game():
    # Initializing the board
    board = np.zeros(NUMBER_OF_SQUARES, dtype="int")
    initial_twos = np.random.default_rng().choice(NUMBER_OF_SQUARES, 2, replace=False)
    board[initial_twos] = 2
    board = board.reshape((CELL_COUNT, CELL_COUNT))
    return board


def push_board_right(board):
    # Pushes all the elements in the board to the right
    new = np.zeros((CELL_COUNT, CELL_COUNT), dtype="int")
    done = False
    for row in range(CELL_COUNT):
        count = CELL_COUNT - 1
        for col in range(CELL_COUNT - 1, -1, -1):
            if board[row][col] != 0:
                new[row][count] = board[row][col]
                if col != count:
                    done = True
                count -= 1
    return new, done


def merge_elements(board):
    # Merges the elements in the board
    score = 0
    done = False
    for row in range(CELL_COUNT):
        for col in range(CELL_COUNT - 1, 0, -1):
            if board[row][col] == board[row][col - 1] and board[row][col] != 0:
                board[row][col] *= 2
                score += board[row][col]
                board[row][col - 1] = 0
                done = True
    return board, done, score


def move_up(board):
    # Moves the board up
    rotated_board = np.rot90(board, -1)
    pushed_board, has_pushed = push_board_right(rotated_board)
    merged_board, has_merged, score = merge_elements(pushed_board)
    second_pushed_board, _ = push_board_right(merged_board)
    rotated_back_board = np.rot90(second_pushed_board)
    move_made = has_pushed or has_merged
    return rotated_back_board, move_made, score


def move_down(board):
    # Moves the board down
    board = np.rot90(board)
    board, has_pushed = push_board_right(board)
    board, has_merged, score = merge_elements(board)
    board, _ = push_board_right(board)
    board = np.rot90(board, -1)
    move_made = has_pushed or has_merged
    return board, move_made, score


def move_left(board):
    # Moves the board left
    board = np.rot90(board, 2)
    board, has_pushed = push_board_right(board)
    board, has_merged, score = merge_elements(board)
    board, _ = push_board_right(board)
    board = np.rot90(board, -2)
    move_made = has_pushed or has_merged
    return board, move_made, score


def move_right(board):
    # Moves the board right
    board, has_pushed = push_board_right(board)
    board, has_merged, score = merge_elements(board)
    board, _ = push_board_right(board)
    move_made = has_pushed or has_merged
    return board, move_made, score


def move(board, direction):
    # Moves the board in the given direction.
    if direction == 0:
        return move_up(board)
    elif direction == 1:
        return move_right(board)
    elif direction == 2:
        return move_down(board)
    elif direction == 3:
        return move_left(board)
    else:
        return board, False, 0


def get_moves():
    # Returns all moves
    return [move_up, move_right, move_down, move_left]


def get_all_possible_moves(board):
    # Returns all possible moves from the board
    all_possible_moves = []
    for direction in range(POSSIBLE_MOVES_COUNT):
        new_board, move_made, _ = move(board, direction)
        if move_made:
            all_possible_moves.append((new_board, direction))
    return all_possible_moves


def fixed_move(board):
    # Moves the board in a fixed order. First move possible is made.
    move_order = [move_left, move_up, move_down, move_right]
    for func in move_order:
        new_board, move_made, _ = func(board)
        if move_made:
            return new_board, True
    return board, False


def random_move(board):
    # Moves the board in a random order. First move possible is made.
    move_made = False
    move_order = [move_right, move_up, move_down, move_left]
    while not move_made and len(move_order) > 0:
        move_index = np.random.randint(0, len(move_order))
        move = move_order[move_index]
        board, move_made, score = move(board)
        if move_made:
            return board, True, score
        move_order.pop(move_index)
    return board, False, score


def get_empty_cells(board):
    # Returns the empty cells in the board
    return np.nonzero(np.logical_not(board))


def add_new_tile(board):
    # Adds a new tile to the board. The tile is either a 2 or a 4 and is placed in a random empty cell.
    # Probability of a 2 is 90% and probability of a 4 is 10%.
    tile_value = NEW_TILE_DISTRIBUTION[np.random.randint(0, len(NEW_TILE_DISTRIBUTION))]
    tile_row_options, tile_col_options = get_empty_cells(board)
    tile_loc = np.random.randint(0, len(tile_row_options))
    board[tile_row_options[tile_loc], tile_col_options[tile_loc]] = tile_value
    return board


def check_for_win(board):
    # Checks if the game is won. The game is won if there is a 2048 tile in the board.
    return 2048 in board


def check_for_loss(board):
    # Checks if the game is lost. The game is lost if there are no possible moves.
    for direction in range(POSSIBLE_MOVES_COUNT):
        new_board, move_made, _ = move(board, direction)
        if move_made:
            return False
    return True


def terminal_state(board):
    # Checks if the game is in a terminal state.
    return check_for_win(board) or check_for_loss(board)


def within_bounds(pos):
    # Checks if the position is within the bounds of the board.
    return (0 <= pos[0] < CELL_COUNT) and (0 <= pos[1] < CELL_COUNT)
