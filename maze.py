from enum import IntEnum
import numpy as np


class Action(IntEnum):
    """Lists all actions that agent can execute."""
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    
    
class Cell(IntEnum):
    """Lists types of cells."""
    AVAILABLE = 0
    WALL = 1
    CURRENT = 2
    START = 3
    EXIT = 4
    
    
class Status(IntEnum):
    """Status of the game."""
    WON = 0
    LOST = 1
    ONGOING = 2


class Maze:
    actions = [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]
    def __init__(self, maze, reward_exit=10.0, penalty_per_step=-0.05, penalty_impossible_move = -0.75):
        self.maze = maze
        self.reward_exit = reward_exit
        self.penalty_per_step = penalty_per_step
        self.penalty_impossible_move = penalty_impossible_move
        self.__loss_threshold = -0.5 * self.maze.size

        self.nrows, self.ncols = self.maze.shape
        self.cells = [(row, col) for col in range(self.ncols) for row in range(self.nrows)]
        self.empty = [(row, col) for col in range(self.ncols) for row in range(self.nrows) if self.maze[row, col] == Cell.AVAILABLE or self.maze[row, col] == Cell.START or self.maze[row, col] == Cell.EXIT]
        self.__exit_cell = (self.ncols - 1, self.nrows - 1)
        self.__start_cell = (0, 0)
        for row in range(self.nrows):
            for col in range(self.ncols):
                if self.maze[row, col] == Cell.START:
                    self.__start_cell = (row, col)
                elif self.maze[row, col] == Cell.EXIT:
                    self.__exit_cell = (row, col)
        self.empty.remove(self.__exit_cell)
        self.reset()
        
    def step(self, action):
        """ Move agent and return new state, reward and game status.
        """
        self.__execute_action(action)
        reward = self.__calculate_reward()
        self.__total_reward += reward
        status = self.__game_status()
        state = self.__current_position()
        return state, reward, status
    
    def reset(self):
        """Reset the game to the starting point.
        """
        self.__previous_cell = self.__start_cell
        self.__current_cell = self.__start_cell
        self.__total_reward = 0.0
        return self.__current_position()

    def __execute_action(self, action):
        """Change the state according to the action."""
        possible_actions = self.__possible_actions()
        if action in possible_actions:
            row, col = self.__current_cell
            if action == Action.LEFT:
                col -= 1
            elif action == Action.UP:
                row -= 1
            if action == Action.RIGHT:
                col += 1
            elif action == Action.DOWN:
                row += 1
            self.__previous_cell = self.__current_cell
            self.__current_cell = (row, col)
          
    def __calculate_reward(self):
        """Returns reward or penalty calculated for executed action."""
        if self.__current_cell == self.__exit_cell:
            reward = self.reward_exit   
        elif self.__current_cell == self.__previous_cell:
            reward = self.penalty_impossible_move
        else:
            reward = self.penalty_per_step  
        return reward

    def __possible_actions(self):
        """Returns a list a permitted actions for current location"""
        row, col = self.__current_cell
        possible_actions = Maze.actions.copy()
        if row == 0 or (row > 0 and self.maze[row - 1, col] == Cell.WALL):
            possible_actions.remove(Action.UP)
        if row == self.nrows - 1 or (row < self.nrows - 1 and self.maze[row + 1, col] == Cell.WALL):
            possible_actions.remove(Action.DOWN)
        if col == 0 or (col > 0 and self.maze[row, col - 1] == Cell.WALL):
            possible_actions.remove(Action.LEFT)
        if col == self.ncols - 1 or (col < self.ncols - 1 and self.maze[row, col + 1] == Cell.WALL):
            possible_actions.remove(Action.RIGHT)
        return possible_actions

    def __game_status(self):
        """ Returns the game status(WON, LOST, ONGOING)."""
        if self.__current_cell == self.__exit_cell:
            return Status.WON
        if self.__total_reward < self.__loss_threshold:
            return Status.LOST
        return Status.ONGOING

    def __current_position(self):
        """Returns current position on the board."""
        return np.array([[self.__current_cell[0], self.__current_cell[1]]])

    

