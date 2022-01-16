from qtable import QTable
from maze import Maze
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    maze_array = np.genfromtxt('maze.txt', delimiter=1, dtype=int)
    print(maze_array)
    my_maze = Maze(maze_array)
    game = QTable(my_maze)
    
    plt.figure(1)
    for epsilon in [0.05, 0.2, 0.50]:
        print(f"epsilon={epsilon}")
        reward_history, state_history, win_history = game.train(exploration_rate=epsilon)
        plt.plot(np.arange(0, len(reward_history), 1), reward_history, label=f"$\epsilon={epsilon}$")
    plt.title("Q-learning algorithm")
    plt.xlabel("Episode number[-]")
    plt.ylabel("Reward [-]")
    plt.ylim([-20, 20])
    plt.legend()
    plt.savefig("q-learning-epsilon.png")
    
    plt.figure(2)
    for alpha in [0.005, 0.05, 0.1]:
        print(f"alpha={alpha}")
        reward_history, state_history, win_history = game.train(learning_rate=alpha)
        label = r"$\alpha=$"+f"{alpha}"
        plt.plot(np.arange(0, len(reward_history), 1), reward_history, label=label)
    plt.title("Q-learning algorithm")
    plt.xlabel("Episode number[-]")
    plt.ylabel("Reward [-]")
    plt.legend()
    plt.savefig("q-learning-alpha.png")
    
    plt.figure(3)
    for gamma in [0.1, 0.9, 0.7]:
        print(f"gamma={gamma}")
        reward_history, state_history, win_history = game.train(discount_rate=gamma)
        plt.plot(np.arange(0, len(reward_history), 1), reward_history, label=f"$\gamma={gamma}$")
    plt.title("Q-learning algorithm")
    plt.xlabel("Episode number[-]")
    plt.ylabel("Reward [-]")
    plt.legend()
    plt.savefig("q-learning-gamma.png")