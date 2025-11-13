from abc import ABC, abstractmethod

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import v0_thin_ice as ti
import matplotlib.patches as mpatches

from typing import Final

#base folders
GRAPHS_FOLDER_NAME: Final[str] = './Graphs_generated/'
PK_FOLDER_NAME: Final[str] = './PK_generated'
POLICY_FOLDER_NAME: Final[str] = './Visualized_Policy_generated'

# An interface for training agents on Thin Ice Environment
class ThinIceTrainingAgent(ABC):
    def __init__(self, env_id: str ='thin-ice-v0', level_str: str ='level_0.txt'):
        self.env_id: str = env_id
        self.level_str: str = level_str
        self.reference_name: str = self.env_id + "-" + self.level_str.split('.')[0]

    @abstractmethod
    def train(self, gamma: float = 0.9, step_size: float = 0.1, epsilon: float = 0.1, n_episodes: int = 1000):
        pass

    @abstractmethod
    def deploy(self, render: bool = True, max_steps: int = 500):
        pass

    def generate_graph(self, steps_per_episode: np.ndarray):
        n_episodes: int = len(steps_per_episode)
        graph_name: str = self.reference_name + '-graph.png'

        sum_steps  = np.zeros(n_episodes)
        for t in range(n_episodes):
            sum_steps[t] = np.mean(steps_per_episode[max(0,t-100):(t+1)]) #avg step
        plt.plot(sum_steps)
        path_for_graph = os.path.join(os.path.dirname(__file__), GRAPHS_FOLDER_NAME,"QLearning", graph_name)
        plt.savefig(path_for_graph)

    def getPkFolderPath(self,algo):
        return os.path.join(os.path.dirname(__file__),PK_FOLDER_NAME,algo)
    
    def getGraphFolderPath(self,algo):
         return os.path.join(os.path.dirname(__file__),GRAPHS_FOLDER_NAME,algo)
    

    def visualize_policy(self):
        env = gym.make(self.env_id, level_str=self.level_str)
        q_path = os.path.join(self.getPkFolderPath("QLearning"), self.reference_name + '_solution.pk1')
        
        with open(q_path, "rb") as f:
            q = pickle.load(f)

        to_cell = env.unwrapped._to_cell  #(x, y, w_mask, avail_mask)
        n_states = env.unwrapped.n_states

        #Get map bounds
        max_x = env.unwrapped.level.n_cols
        max_y = env.unwrapped.level.n_rows

        #Prepare plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-0.5, max_x - 0.5)
        ax.set_ylim(-0.5, max_y - 0.5)
        ax.set_aspect('equal')

        visited_tiles = set()
        state = env.reset()[0]
        terminated = False
        step_count = 0
        max_steps = 500  #prevent infinite loop

        #Get visited tiles to draw arrows on plot
        while not terminated and step_count < max_steps:
            step_count += 1
            x, y, _ = to_cell[state]
            visited_tiles.add((x, y))
            action = np.argmax(q[state])
            state, reward, terminated, _, _ = env.step(action)
            env.close()

        start_pos = env.unwrapped.level.player_start
        goal_pos = env.unwrapped.level.target 

        best_action_per_cell = {}

        for state in range(n_states):
            x, y, avail_mask = to_cell[state]

            if avail_mask == 0: #Skip blocked cells
                continue

            best_action = np.argmax(q[state])
            best_value = np.max(q[state])

            #Only keep the highest Q-value for each (x, y)
            if (x, y) not in best_action_per_cell or best_value > best_action_per_cell[(x, y)][1]:
                best_action_per_cell[(x, y)] = (best_action, best_value)

        #Get tiles for walls
        for row in env.unwrapped.level.tiles:
            for tile in row:
                if tile.tile_type in (ti.LevelTileType.WALL, ti.LevelTileType.BLANK):
                    tile_x, tile_y = tile.position
                    plot_y = max_y - tile_y - 1
                    ax.add_patch(plt.Rectangle((tile_x - 0.5, plot_y - 0.5), 1, 1, color='black')) # Draw obstacle

        for (x, y), (best_action, _) in best_action_per_cell.items():

            if visited_tiles and (x, y) not in visited_tiles:
                continue

            plot_y = max_y - y - 1

            #Draw start/goal
            if (x, y) == start_pos:
                ax.add_patch(plt.Rectangle((x - 0.5, plot_y - 0.5), 1, 1, color='yellow'))
            elif (x, y) == goal_pos:
                ax.add_patch(plt.Rectangle((x - 0.5, plot_y - 0.5), 1, 1, color='green'))

            #Draw arrow for best action
            dx, dy = 0, 0
            if best_action == 0:   # left
                dx = -0.3
            elif best_action == 1: # down
                dy = -0.3
            elif best_action == 2: # right
                dx = 0.3
            elif best_action == 3: # up
                dy = 0.3

            ax.arrow(x, plot_y, dx, dy, head_width=0.2, color='blue', length_includes_head=True)

        cov_patch = mpatches.Patch(color='blue', label='Policy Path')
        start_patch = mpatches.Patch(color='yellow', label='Start Pos')
        goal_patch = mpatches.Patch(color='green', label='End Pos')
        obs_patch = mpatches.Patch(color='black', label='Walls/Blanks')

        plt.legend(handles=[cov_patch, start_patch, goal_patch, obs_patch], loc='upper right')
        plt.title(f"Optimal Policy Path - {self.level_str}")
        ax.set_xticks(range(max_x))
        ax.set_yticks(range(max_y))
        ax.grid(True)
        policy_name = self.reference_name + "policy-plot-generated.png"
        path_for_policy = os.path.join(os.path.dirname(__file__), POLICY_FOLDER_NAME, "QLearning", policy_name)
        plt.savefig(path_for_policy)
        # plt.show()