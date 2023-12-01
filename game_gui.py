from tkinter import Frame, Label, CENTER

import AI_HW3.game_ai as game_ai
import AI_HW3.game_functions as game_functions

EDGE_LENGTH = 400
CELL_COUNT = 4
CELL_PAD = 10

UP_KEY = "'w'"
DOWN_KEY = "'s'"
LEFT_KEY = "'a'"
RIGHT_KEY = "'d'"
EXPECTIMAX_AGENT = "'e'"
SIMPLE_MCTS_AGENT = "'m'"
UCB_MCTS_AGENT = "'u'"

LABEL_FONT = ("Verdana", 40, "bold")


def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


GAME_COLOR = "#C0B3A9"

EMPTY_COLOR = rgb_to_hex(204, 192, 179)

TILE_COLORS = {2: rgb_to_hex(238, 228, 218), 4: rgb_to_hex(237, 224, 200),
               8: rgb_to_hex(242, 177, 121), 16: rgb_to_hex(245, 149, 99), 32: rgb_to_hex(246, 124, 95),
               64: rgb_to_hex(246, 94, 59), 128: rgb_to_hex(237, 207, 114), 256: rgb_to_hex(237, 204, 97),
               512: rgb_to_hex(237, 200, 80), 1024: rgb_to_hex(237, 197, 63), 2048: rgb_to_hex(237, 194, 46), }

LABEL_COLORS = {2: rgb_to_hex(119, 110, 101), 4: rgb_to_hex(119, 110, 101), 8: rgb_to_hex(119, 110, 101),
                16: rgb_to_hex(249, 246, 242), 32: rgb_to_hex(249, 246, 242), 64: rgb_to_hex(249, 246, 242),
                128: rgb_to_hex(249, 246, 242), 256: rgb_to_hex(249, 246, 242), 512: rgb_to_hex(249, 246, 242),
                1024: rgb_to_hex(249, 246, 242), 2048: rgb_to_hex(249, 246, 242)}


class Display(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_press)

        self.commands = {UP_KEY: game_functions.move_up,
                         DOWN_KEY: game_functions.move_down,
                         LEFT_KEY: game_functions.move_left,
                         RIGHT_KEY: game_functions.move_right,
                         }

        self.grid_cells = []
        self.build_grid()
        self.init_matrix()
        self.draw_grid_cells()

        self.mainloop()

    def build_grid(self):
        background = Frame(self, bg=GAME_COLOR,
                           width=EDGE_LENGTH, height=EDGE_LENGTH)
        background.grid()

        for row in range(CELL_COUNT):
            grid_row = []
            for col in range(CELL_COUNT):
                cell = Frame(background, bg=EMPTY_COLOR,
                             width=EDGE_LENGTH / CELL_COUNT,
                             height=EDGE_LENGTH / CELL_COUNT)
                cell.grid(row=row, column=col, padx=CELL_PAD,
                          pady=CELL_PAD)
                t = Label(master=cell, text="",
                          bg=EMPTY_COLOR,
                          justify=CENTER, font=LABEL_FONT, width=5, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def init_matrix(self):
        self.matrix = game_functions.initialize_game()

    def draw_grid_cells(self):
        for row in range(CELL_COUNT):
            for col in range(CELL_COUNT):
                tile_value = self.matrix[row][col]
                if not tile_value:
                    self.grid_cells[row][col].configure(
                        text="", bg=EMPTY_COLOR)
                else:
                    self.grid_cells[row][col].configure(text=str(
                        tile_value), bg=TILE_COLORS[tile_value],
                        fg=LABEL_COLORS[tile_value])
        self.update_idletasks()

    def key_press(self, event):
        valid_game = True
        key = repr(event.char)

        if key == SIMPLE_MCTS_AGENT or key == EXPECTIMAX_AGENT or key == UCB_MCTS_AGENT:
            move_count = 0
            while valid_game:
                if key == SIMPLE_MCTS_AGENT:
                    self.matrix, valid_game = game_ai.ai_move(
                        self.matrix, move_count, 'mcts')
                elif key == EXPECTIMAX_AGENT:
                    self.matrix, valid_game = game_ai.ai_move(
                        self.matrix, move_count)
                elif key == UCB_MCTS_AGENT:
                    self.matrix, valid_game = game_ai.ai_move(
                        self.matrix, move_count, 'ucb')
                if valid_game:
                    self.matrix = game_functions.add_new_tile(self.matrix)
                    self.draw_grid_cells()
                move_count += 1
        elif key in self.commands:
            self.matrix, move_made, _ = self.commands[repr(
                event.char)](self.matrix)
            if move_made:
                self.matrix = game_functions.add_new_tile(self.matrix)
                self.draw_grid_cells()
                move_made = False


gamegrid = Display()