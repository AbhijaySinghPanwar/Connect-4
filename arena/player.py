from arena.llm import LLM
from arena.board import pieces, cols
import json
import random
import logging


class Player:
    """
    This class represents one AI player in the game, and is responsible for managing the prompts
    Delegating to an LLM instance to connect to the LLM
    """

    def __init__(self, model: str, color: int):
        """
        Set up this instance for the given model and player color
        """
        self.color = color
        self.model = model
        self.llm = LLM.create(self.model)
        self.evaluation = ""
        self.threats = ""
        self.opportunities = ""
        self.strategy = ""

    def system(self, board, legal_moves: str, illegal_moves: str) -> str:
        """
        Return the system prompt for this move
        """
        return f"""You are playing the board game Connect 4.
Players take turns to drop counters into one of 7 columns A, B, C, D, E, F, G.
The winner is the first player to get 4 counters in a row in any direction.
You are {pieces[self.color]} and your opponent is {pieces[self.color * -1]}.
You must pick a column for your move. You must pick one of the following legal moves: {legal_moves}.
You should respond in JSON according to this spec:

{{
    "evaluation": "my assessment of the board (one or two short sentences)",
    "threats": "any threats from my opponent that I should block (one or two short sentences)",
    "opportunities": "my best chances to win (one or two short sentences)",
    "strategy": "my thought process (one or two short sentences)",
    "move_column": "one letter from this list of legal moves: {legal_moves}"
}}

You must pick one of these letters for your move_column: {legal_moves}{illegal_moves}"""

    def user(self, board, legal_moves: str, illegal_moves: str) -> str:
        """
        Return the user prompt for this move
        """
        return f"""It is your turn to make a move as {pieces[self.color]}.
Here is the current board, with row 1 at the bottom of the board:

{board.json()}

Here's another way of looking at the board visually, where R represents a red counter, Y for a yellow counter, and _ represents an empty square.

{board.alternative()}

Your final response should be only in JSON strictly according to this spec:

{{
    "evaluation": "my assessment of the board (one or two short sentences)",
    "threats": "any threats from my opponent that I should block (one or two short sentences)",
    "opportunities": "my best chances to win (one or two short sentences)",
    "strategy": "my thought process (one or two short sentences)",
    "move_column": "one of {legal_moves} which are the legal moves"
}}

Keep each reasoning field to at most two short sentences.

For example, the following could be a response:

{{
    "evaluation": "the board is equally balanced but I have a slight advantage",
    "threats": "my opponent has a threat but I can block it",
    "opportunities": "I've developed several promising 3 in a row opportunities",
    "strategy": "I must first block my opponent, then I can continue to develop",
    "move_column": "{random.choice(board.legal_moves())}"
}}

And this is another example of a well formed response:

{{
    "evaluation": "although my opponent has more threats, I can win immediately",
    "threats": "my opponent has several threats",
    "opportunities": "I can immediately win the game by making a diagonal 4",
    "strategy": "I will take the winning move",
    "move_column": "{random.choice(board.legal_moves())}"
}}


Now make your decision.
You must pick one of these letters for your move_column: {legal_moves}{illegal_moves}
"""

    def process_move(self, reply: str, board) -> bool:
        """
        Interpret the reply and make the move; if the move is illegal, then the current player loses.

        Move parsing rules (evidence from audit Step 5, Failures 3 and 4):
        - Accept "D", "d", "D ", " D"  (strip whitespace, uppercase)
        - Reject "Column D", "Move D", "" or any multi-character value
          (cols.find returns -1, caught by the legality check)
        - Never silently guess; an ambiguous or verbose value raises ValueError
        """
        try:
            logging.info(f"[Player {self.color}] Raw reply from LLM: {reply[:300]}")

            # Minimal single-letter shortcut: handle {D} style responses
            if len(reply) == 3 and reply[0] == "{" and reply[2] == "}":
                reply = f'{{"move_column": "{reply[1]}"}}'

            result = json.loads(reply)

            # Extract and normalise move_column
            move_raw = str(result.get("move_column") or "")
            move = move_raw.strip().upper()

            logging.info(
                f"[Player {self.color}] move_column raw={move_raw!r} normalised={move!r}"
            )

            # Require exactly one character that maps to a valid column (A–G).
            # Multi-word values such as "Column D" produce len > 1 → col = -1.
            # Blank or missing values produce len == 0 → col = -1.
            if len(move) == 1 and move in cols:
                col = cols.find(move)
            else:
                col = -1
                logging.error(
                    f"[Player {self.color}] move_column={move_raw!r} is not a single "
                    f"valid column letter — rejected"
                )

            if not (0 <= col <= 6) or board.height(col) == 6:
                raise ValueError(
                    f"Illegal move: move_column={move_raw!r}, col={col}"
                )

            board.move(col)
            self.evaluation = result.get("evaluation") or ""
            self.threats = result.get("threats") or ""
            self.opportunities = result.get("opportunities") or ""
            self.strategy = result.get("strategy") or ""
            return True

        except Exception as e:
            logging.error(f"[Player {self.color}] Exception processing move: {e}")
            return False

    def move(self, board):
        """
        Have the underlying LLM make a move, and process the result with 1 retry for illegal moves
        """
        legal_moves = ", ".join(board.legal_moves())
        if illegal := board.illegal_moves():
            illegal_moves = (
                "\nYou must NOT make any of these moves which are ILLEGAL: "
                + ", ".join(illegal)
            )
        else:
            illegal_moves = ""
            
        system = self.system(board, legal_moves, illegal_moves)
        user = self.user(board, legal_moves, illegal_moves)
        
        retries = 1
        attempts = 0
        
        while attempts <= retries:
            attempts += 1
            reply = self.llm.send(system, user)
            success = self.process_move(reply, board)
            if success:
                return
            else:
                logging.warning(
                    f"Player {self.color} attempt {attempts} resulted in an illegal move or error."
                )
                if attempts <= retries:
                    logging.info(f"Retrying move for Player {self.color}...")
                    
        # If we fail after all retries, forfeit
        logging.error(
            f"Player {self.color} failed to make a legal move after "
            f"{retries + 1} attempts. Forfeiting."
        )
        board.forfeit = True
        board.winner = -1 * board.player

    def thoughts(self):
        """
        Return HTML to describe the inner thoughts
        """
        result = '<div style="text-align: left;font-size:14px"><br/>'
        result += f"<b>Evaluation:</b><br/>{self.evaluation}<br/><br/>"
        result += f"<b>Threats:</b><br/>{self.threats}<br/><br/>"
        result += f"<b>Opportunities:</b><br/>{self.opportunities}<br/><br/>"
        result += f"<b>Strategy:</b><br/>{self.strategy}"
        result += "</div>"
        return result

    def switch_model(self, new_model_name: str):
        """
        Change the underlying LLM to the new model
        """
        self.llm = LLM.create(new_model_name)
