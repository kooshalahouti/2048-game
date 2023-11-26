import time
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

from game_ai import ai_move
import game_functions as gf

SAMPLE_COUNT = 10  # Number of games to play in the plot (Higher = more accurate but slower)


def ai_play(board, agent_name='expectimax'):
    move_number = 0
    valid_game = True
    while valid_game:
        move_number += 1
        board, valid_game = ai_move(board, move_number, agent_name=agent_name)
        if valid_game:
            board = gf.add_new_tile(board)
        if gf.terminal_state(board):
            valid_game = False
        if move_number % 20 == 0 or move_number < 10:
            pprint(board)
            print(move_number)
    pprint(board)
    return np.amax(board)


def ai_plot(agent_name='expectimax'):
    tick_locations = np.arange(1, 12)
    final_scores = []
    for i in range(SAMPLE_COUNT):
        print('round is ', i)
        np.random.seed(3)
        board = gf.initialize_game()
        max_value = ai_play(board, agent_name=agent_name)
        final_scores.append(max_value)
    print(final_scores)
    all_counts = np.zeros(11)
    unique, counts = np.unique(np.array(final_scores), return_counts=True)
    unique = np.log2(unique).astype(int)
    index = 0

    for tick in tick_locations:
        if tick in unique:
            all_counts[tick - 1] = counts[index]
            index += 1

    plt.bar(tick_locations, all_counts)
    plt.xticks(tick_locations, np.power(2, tick_locations))
    plt.xlabel("Score of Game", fontsize=24)
    plt.ylabel(f"Frequency per {SAMPLE_COUNT} runs", fontsize=24)
    plt.show()
    print(np.mean(final_scores))


# If you want to run this file, comment out the following line.

# 1. playing game once
# board = gf.initialize_game()
# ai_play(board, agent_name='mcts')

# 2. plotting
tik = time.time()
ai_plot(agent_name='expectimax')
tok = time.time()
print(tok - tik)
print((tok - tik) / SAMPLE_COUNT)
