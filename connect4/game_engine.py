class Connect4Game:
    """Connect 4 game engine"""
    
    def __init__(self):
        """Initialize a new Connect 4 game"""
        self.reset_game()
    
    def reset_game(self):
        """Reset the game to initial state"""
        # Board is 7 columns x 6 rows
        # 0 = empty, 1 = player 1 (red), 2 = player 2 (yellow)
        self.board = [[0 for _ in range(6)] for _ in range(7)]
        self.current_player = 1  # Player 1 starts
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.moves_count = 0
    
    def make_move(self, column):
        """
        Make a move in the specified column
        
        Args:
            column: Column index (0-6)
            
        Returns:
            bool: True if move was valid, False otherwise
        """
        if self.game_over or column < 0 or column > 6:
            return False
        
        # Find the first empty cell in the column (from bottom to top)
        for row in range(5, -1, -1):
            if self.board[column][row] == 0:
                self.board[column][row] = self.current_player
                self.last_move = (column, row)
                self.moves_count += 1
                
                # Check for win
                if self._check_win(column, row):
                    self.game_over = True
                    self.winner = self.current_player
                # Check for draw
                elif self.moves_count == 42:  # 7*6 = 42 cells
                    self.game_over = True
                    self.winner = 0  # Draw
                else:
                    # Switch player
                    self.current_player = 3 - self.current_player  # 1 -> 2, 2 -> 1
                
                return True
        
        # Column is full
        return False
    
    def _check_win(self, col, row):
        """Check if the last move resulted in a win"""
        player = self.board[col][row]
        directions = [
            [(0, 1), (0, -1)],  # Vertical
            [(1, 0), (-1, 0)],  # Horizontal
            [(1, 1), (-1, -1)],  # Diagonal /
            [(1, -1), (-1, 1)]   # Diagonal \
        ]
        
        for dir_pair in directions:
            count = 1  # Count the piece just placed
            
            # Check both directions
            for dx, dy in dir_pair:
                x, y = col, row
                
                # Count consecutive pieces in this direction
                for _ in range(3):  # Need 3 more to make 4 in a row
                    x += dx
                    y += dy
                    
                    if (0 <= x < 7 and 0 <= y < 6 and 
                            self.board[x][y] == player):
                        count += 1
                    else:
                        break
            
            if count >= 4:
                return True
        
        return False
    
    def get_valid_moves(self):
        """Return list of valid columns to play"""
        if self.game_over:
            return []
        
        return [col for col in range(7) if self.board[col][0] == 0]
    
    def get_game_state(self):
        """Return the current game state as a dictionary"""
        return {
            "board": self.board,
            "current_player": self.current_player,
            "game_over": self.game_over,
            "winner": self.winner,
            "last_move": self.last_move,
            "valid_moves": self.get_valid_moves()
        }
    
    def get_player_llm_prompt(self):
        """Generate a prompt for the LLM to make a move"""
        prompt = f"""
You are playing Connect 4 against another player. Here's the current game state:

Current board (0=empty, 1=red, 2=yellow):
"""
        # Display board (rotated for easier visualization)
        for row in range(6):
            row_str = "|"
            for col in range(7):
                cell = self.board[col][row]
                if cell == 0:
                    row_str += " "
                elif cell == 1:
                    row_str += "R"
                else:
                    row_str += "Y"
                row_str += "|"
            prompt += row_str + "\n"
        
        prompt += "+-+-+-+-+-+-+-+\n"
        prompt += "|0|1|2|3|4|5|6|\n\n"
        
        prompt += f"You are {'red (R)' if self.current_player == 1 else 'yellow (Y)'} (player {self.current_player}).\n\n"
        
        prompt += """Please choose a column (0-6) to drop your piece.

You can use <think>...</think> tags to show your reasoning process.

Respond with ONLY a single digit (0-6) representing your chosen column.
"""
        return prompt
    
    def process_player_llm_response(self, response):
        """
        Process the LLM's response and make a move
        
        Args:
            response: The LLM's response text
            
        Returns:
            dict: Result of the move
        """
        # Extract thinking content if present
        thinking = None
        clean_response = response
        
        if "<think>" in response and "</think>" in response:
            import re
            think_match = re.search(r"<think>([\s\S]*?)</think>", response)
            if think_match:
                thinking = think_match.group(1).strip()
                clean_response = response.replace(think_match.group(0), "").strip()
        
        # Try to extract a column number from the response
        import re
        column_match = re.search(r"[0-6]", clean_response)
        
        if column_match:
            column = int(column_match.group(0))
            success = self.make_move(column)
            
            return {
                "column": column,
                "success": success,
                "thinking": thinking,
                "game_state": self.get_game_state()
            }
        else:
            return {
                "column": None,
                "success": False,
                "thinking": thinking,
                "error": "Invalid response. Could not extract column number.",
                "game_state": self.get_game_state()
            }