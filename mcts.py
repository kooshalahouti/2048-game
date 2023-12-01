import numpy as np
import AI_HW3.evaluation as evaluation
import AI_HW3.game_functions as gf

class MCTS:
    def __init__(self, board, mode='ucb'):
        self.C_CONSTANT = 2
        self.SD_SCALE_PARAM = 5
        self.TM_SCALE_PARAM = 5
        self.SCALER_PARAM = 200
        self.UCB_SD_SCALE_PARAM = 5
        self.UCB_TM_SCALE_PARAM = 5
        self.UCB_SCALER_PARAM = 300
        self.board = board
        self.mode = mode

    def get_search_params(self, move_number: int) -> (int, int):
        base_value = 100

        if self.mode == 'ucb':
            sd_scale = self.UCB_SD_SCALE_PARAM
            tm_scale = self.UCB_TM_SCALE_PARAM
            scaler = self.UCB_SCALER_PARAM
        else:
            sd_scale = self.SD_SCALE_PARAM
            tm_scale = self.TM_SCALE_PARAM
            scaler = self.SCALER_PARAM

        search_depth = base_value + move_number * sd_scale
        total_moves = base_value + move_number * tm_scale

        return search_depth, total_moves

    def simulate_move(self, board: np.ndarray, search_depth: int) -> float:
        total_score = 0.0
        original_board = np.copy(board)
        temp_board = np.copy(original_board)

        for _ in range(search_depth):
            np.copyto(temp_board, original_board)  # Reuse the same board
            depth_search = np.random.randint(1, search_depth + 1)

            for _ in range(depth_search):
                moves = gf.get_moves(temp_board)
                if moves:
                    move = np.random.choice(moves)
                    move_made = gf.make_move(temp_board, move)

                    if move_made:
                        total_score += evaluation.evaluate_state(temp_board)
                        gf.add_new_tile(temp_board)

        return total_score

    def ucb(self, moves: list, total_visits: int) -> np.ndarray:
        ucb_scores = np.zeros(len(moves))

        for i, move in enumerate(moves):
            action, score, visits = move
            exploration_term = self.C_CONSTANT * np.sqrt(np.log(total_visits) / (visits + 1))
            ucb_scores[i] = score / (visits + 1) + exploration_term

        return ucb_scores

    def mcts_v0(self, board: np.ndarray, total_moves: int, search_depth: int):
        moves = gf.get_moves(board)
        scores = np.zeros(len(moves))

        for i, action in enumerate(moves):
            for _ in range(total_moves):
                temp_board = np.copy(board)
                gf.make_move(temp_board, action)
                scores[i] += self.simulate_move(temp_board, search_depth)

        ucb_scores = self.ucb([(m, score, 1) for m, score in zip(moves, scores)], total_moves)
        best_move_index = np.argmax(ucb_scores)
        return moves[best_move_index]

    def mcts_v2(self, board, total_moves, search_depth):
        moves_data = [(action, 0, 0) for action in gf.get_moves(board)]

        for _ in range(total_moves):
            ucb_scores = self.ucb(moves_data, sum(x[2] for x in moves_data))
            best_move_index = np.argmax(ucb_scores)
            best_move, _, _ = moves_data[best_move_index]

            temp_board = np.copy(board)
            move_made = gf.make_move(temp_board, best_move)

            if move_made:
                score = self.simulate_move(temp_board, search_depth)
                moves_data[best_move_index] = (best_move, moves_data[best_move_index][1] + score, moves_data[best_move_index][2] + 1)

        ucb_scores = self.ucb(moves_data, total_moves)
        best_move_index = np.argmax(ucb_scores)
        return moves_data[best_move_index][0]
