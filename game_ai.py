from mcts import MCTS
from expectimax import Expectimax


def get_agent(board, agent_name):
    if agent_name == 'expectimax':
        return Expectimax(board)
    elif agent_name == 'mcts':
        return MCTS(board, 'mcts')
    elif agent_name == 'ucb':
        return MCTS(board, 'ucb')
    else:
        raise ValueError('Invalid agent')


def ai_move(board, move_number, agent_name='expectimax'):
    agent = get_agent(board, agent_name)
    best_move = None
    while best_move is None:
        best_move = agent.ai_move(board, move_number)
    search_board, game_valid, score = best_move(board)
    return search_board, game_valid
