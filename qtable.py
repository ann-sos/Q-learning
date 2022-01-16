import random
import numpy as np
from maze import Status



class QTable():

    def __init__(self, game):
        self.Q = dict()
        self.maze = game

    def train(self, episodes=1000, learning_rate=0.1, exploration_decay=0.995, exploration_rate=0.1, discount_rate=0.9):
        reward_history = []
        win_history = []
        state_history = []
        for episode in range(1, episodes + 1):
            episode_reward = 0
            state = self.maze.reset()
            state = tuple(state.flatten())
            episode_state_history = [state]
            while True:
                # decaying epsilon greedy policy
                # Choose an action for that state based on one of the action selection policies
                if np.random.random() < exploration_rate:
                    action = random.choice(self.maze.actions)
                else:
                    action = self.policy_choose(state)
                # Take the action, and observe the reward and the new state
                next_state, reward, status = self.maze.step(action)
                next_state = tuple(next_state.flatten())

                episode_reward += reward
                # Initialize Q value for the stace, action pair if doesn't already exists
                if (state, action) not in self.Q.keys():
                    self.Q[(state, action)] = 0.0
                #Calculate maximum reward possible for the next state
                maximum_possible_Q = max([self.Q.get((next_state, a), 0.0) for a in self.maze.actions])
                #Update the Q-value for the state using the observed reward and the maximum reward possible for the next state
                self.Q[(state, action)] += learning_rate * (reward + discount_rate * maximum_possible_Q - self.Q[(state, action)])

                if status in (Status.WON, Status.LOST):
                    break
                # Set the state to the new state, and repeat the process until a terminal state is reached
                state = next_state
                episode_state_history.append(state)


            reward_history.append(episode_reward)
            state_history.append(episode_state_history)
            #print(f"episode: {episode} --- score: {status.name} --- reward: {episode_reward}")
            

            exploration_rate *= exploration_decay
        print("Path from start to exit:")
        [print(state, end='->') for state in episode_state_history]
        return reward_history, state_history, win_history
    
    def policy_choose(self, state):
        """Returns action index corresponding to maze.actions"""
        possible_rewards = self.possible_rewards(state)
        actions = np.argmax(possible_rewards)
        return actions
    
    def possible_rewards(self, state):
        """ Get q values for all actions for a certain state. """
        if type(state) == np.ndarray:
            state = tuple(state.flatten())
        rewards = []
        for action in self.maze.actions:
            rewards.append(self.Q.get((state, action), 0.0))
        return np.array(rewards)


