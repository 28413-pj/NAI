"""
=========================================================
                    Dominacja na siatce
=========================================================

   Environment setup:

    Make sure you have Python 3.8+ installed.
     You can check with:
         python --version

     Install the required library:
         pip install easyAI

     Run the game:
         python dominacja_na_siatce.py

   Rules:
    - The board starts empty.
    - Players (X and O) take turns placing their marks on empty cells.
    - You can place a mark only on a cell that is orthogonally adjacent
        (up, down, left, or right) to one of your own existing marks.
    - The first move of each player can be anywhere on the board.
    - If a player has no legal moves, they lose.

   Authors:
    - Marek Jenczyk
    - Oskar Skomra

=========================================================
"""

from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax

ROWS, COLUMNS = 5, 5  # dimensions of the board


class DominacjaNaSiatce(TwoPlayerGame):

    def __init__(self, players):
        """
        Initialize the game.

        Args:
            players (list): A list containing two players
                            [Human_Player(), AI_Player()].
        """
        self.players = players
        self.board = ['.'] * (ROWS * COLUMNS)  # empty board represented as a list
        self.current_player = 1  # player 1 (human) starts

    def possible_moves(self):
        """
        Determine all legal moves for the current player.

        Returns:
            list[str]: A list of valid move positions (1-based indices as strings).
        """
        symbol = 'X' if self.current_player == 1 else 'O'
        first_move = not any(s == symbol for s in self.board)  # check if it's the first move
        moves = []

        for i, c in enumerate(self.board):
            if c != '.':  # only empty cells are valid
                continue

            if first_move:
                # if first move, any empty cell is allowed
                moves.append(str(i + 1))
                continue

            neighbors = []

            # left neighbor
            if i % COLUMNS != 0:
                neighbors.append(i - 1)
            # right neighbor
            if i % COLUMNS != COLUMNS - 1:
                neighbors.append(i + 1)
            # top neighbor
            if i >= COLUMNS:
                neighbors.append(i - COLUMNS)
            # bottom neighbor
            if i < COLUMNS * (ROWS - 1):
                neighbors.append(i + COLUMNS)

            # if any neighbor belongs to the player, it's a valid move
            if any(self.board[n] == symbol for n in neighbors):
                moves.append(str(i + 1))

        return moves

    def make_move(self, move):
        """
        Place the current player's symbol on the chosen cell.

        Args:
            move (str or int): The position on the board (1-based index).
        """
        move = int(move) - 1
        self.board[move] = 'X' if self.current_player == 1 else 'O'

    def unmake_move(self, move):
        """
        Undo a move (used by the AI during simulation).

        Args:
            move (str or int): The position to clear (1-based index).
        """
        self.board[int(move) - 1] = '.'

    def is_over(self):
        """
        Check if the game is over (no possible moves).

        Returns:
            bool: True if the current player has no legal moves, False otherwise.
        """
        return self.possible_moves() == []

    def win(self):
        """
        Determine which player won.

        Returns:
            int or None: 1 if Player 1 (human) won,
                         2 if Player 2 (AI) won,
                         None if the game is still ongoing.
        """
        return (3 - self.current_player) if self.is_over() else None

    def scoring(self):
        """
        Evaluate the current position for the AI.

        Returns:
            int: -100 if the current player has no moves (losing position),
                 otherwise 0.
        """
        return -100 if self.is_over() else 0

    def show(self):
        """
        Display the current state of the board in the console.
        """
        for y in range(COLUMNS):
            row = self.board[y * ROWS:(y + 1) * ROWS]
            print(' '.join(row))


if __name__ == "__main__":
    # Create AI with a search depth of 10 (higher = stronger AI)
    ai = Negamax(10)
    game = DominacjaNaSiatce([Human_Player(), AI_Player(ai)])

    print("\nDominacja na siatce: place your symbols on empty squares adjacent to your own.")
    print("Enter cell numbers (1–25). The first move for each player can be anywhere.\n")

    game.play()

    # After the game ends, print the result
    if game.win() == 1:
        print("\n🏆 Human Player (X) wins!")
    elif game.win() == 2:
        print("\n🤖 AI Player (O) wins!")



