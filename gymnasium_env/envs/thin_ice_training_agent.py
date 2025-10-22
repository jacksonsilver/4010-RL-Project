from abc import ABC, abstractmethod

import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Final

GRAPHS_FOLDER_NAME: Final[str] = './graphs_generated/'
PK_FOLDER_NAME: Final[str] = './pk_files_generated'

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
    def deploy(self, render: bool = True):
        pass

    def generate_graph(self, steps_per_episode: np.ndarray):
        n_episodes: int = len(steps_per_episode)
        graph_name: str = self.reference_name + '-graph.png'

        sum_steps  = np.zeros(n_episodes)
        for t in range(n_episodes):
            sum_steps[t] = np.mean(steps_per_episode[max(0,t-100):(t+1)]) #avg step
        plt.plot(sum_steps)
        path_for_graph = os.path.join(os.path.dirname(__file__), GRAPHS_FOLDER_NAME, graph_name)
        plt.savefig(path_for_graph)

    def getPkFolderPath(self):
        return os.path.join(os.path.dirname(__file__),PK_FOLDER_NAME)