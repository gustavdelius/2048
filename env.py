import numpy as np
import random

class Game2048Env:
    """
    A 3x3 version of the 2048 game compatible with standard RL training loops.
    """
    def __init__(self):
        self.size = 3
        self.action_space_n = 4  # 0: Up, 1: Down, 2: Left, 3: Right
        self.reset()
        
    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.add_random_tile()
        self.add_random_tile()
        return self.get_state()
        
    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if not empty_cells:
            return
        cell = random.choice(empty_cells)
        # 90% chance of 2, 10% chance of 4
        self.board[cell] = 2 if random.random() < 0.9 else 4
        
    def get_state(self):
        # Index encoding of the board for embedding layer
        # 0 -> 0, 2 -> 1, 4 -> 2, 8 -> 3, ...
        # Shape will be (size, size)
        state = np.zeros((self.size, self.size), dtype=np.int64)
        for i in range(self.size):
            for j in range(self.size):
                val = self.board[i, j]
                if val > 0:
                    idx = int(np.log2(val))
                    state[i, j] = min(idx, 8)
        return state
        
    def step(self, action):
        """
        Executes an action in the environment.
        Action mapping: 0=Up, 1=Down, 2=Left, 3=Right
        Returns: next_state, reward, done, info
        """
        # Apply move logic and compute reward
        reward = 0
        changed = False
        
        if action == 0: # Up
            self.board, reward, changed = self.slide_and_merge(self.board, direction='up')
        elif action == 1: # Down
            self.board, reward, changed = self.slide_and_merge(self.board, direction='down')
        elif action == 2: # Left
            self.board, reward, changed = self.slide_and_merge(self.board, direction='left')
        elif action == 3: # Right
            self.board, reward, changed = self.slide_and_merge(self.board, direction='right')
            
        # Give a small negative reward for invalid moves to encourage learning valid moves
        if not changed:
            reward = -1 
        else:
            self.add_random_tile()
            
        done = self.is_game_over()
        
        info = {
            'highest_tile': np.max(self.board),
            'board': self.board.copy(),
            'valid_move': changed
        }
        
        return self.get_state(), float(reward), done, info
        
    def slide_and_merge(self, board, direction):
        # We handle up, down, right by rotating the board to simulate left movements
        if direction == 'left':
            rotated = board
        elif direction == 'right':
            rotated = np.rot90(board, 2)
        elif direction == 'up':
            rotated = np.rot90(board, 1)
        elif direction == 'down':
            rotated = np.rot90(board, 3)
            
        new_board = np.zeros_like(rotated)
        reward = 0
        
        for i in range(self.size):
            row = rotated[i]
            # 1. Slide to remove zeros
            non_zero = row[row != 0]
            
            # 2. Merge adjacent identical elements
            merged_row = []
            skip = False
            for j in range(len(non_zero)):
                if skip:
                    skip = False
                    continue
                if j + 1 < len(non_zero) and non_zero[j] == non_zero[j+1]:
                    new_val = non_zero[j] * 2
                    merged_row.append(new_val)
                    reward += new_val
                    skip = True
                else:
                    merged_row.append(non_zero[j])
                    
            # 3. Slide again into new row
            merged_row = np.array(merged_row)
            new_board[i, :len(merged_row)] = merged_row
            
        # Rotate back to original orientation
        if direction == 'left':
            final_board = new_board
        elif direction == 'right':
            final_board = np.rot90(new_board, -2)
        elif direction == 'up':
            final_board = np.rot90(new_board, -1)
        elif direction == 'down':
            final_board = np.rot90(new_board, -3)
            
        changed = not np.array_equal(board, final_board)
        return final_board, reward, changed

    def is_game_over(self):
        # Empty cell? game not over
        if np.any(self.board == 0):
            return False
            
        # Any adjacent identical pairs horizontally?
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
                    
        # ... or vertically?
        for i in range(self.size - 1):
            for j in range(self.size):
                if self.board[i, j] == self.board[i+1, j]:
                    return False
                    
        return True

    def get_valid_actions(self):
        """Returns a list of valid actions (0: Up, 1: Down, 2: Left, 3: Right)"""
        valid_actions = []
        for a, direction in enumerate(['up', 'down', 'left', 'right']):
            # slide_and_merge does NOT modify the original array if we pass a copy, 
            # wait.. slide_and_merge modifies in place? No, it returns a new array.
            _, _, changed = self.slide_and_merge(self.board, direction)
            if changed:
                valid_actions.append(a)
        return valid_actions
