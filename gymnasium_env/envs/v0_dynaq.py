import gymnasium as gym
import numpy as np
import pickle
import v0_thin_ice_env as ti #including it so it registers 
import os

from thin_ice_training_agent import ThinIceTrainingAgent

class ThincIceDynaQAgent(ThinIceTrainingAgent):
    def __init__(self, env_id: str ='thin-ice-v0', level_str: str ='Level0.txt'):
        super().__init__(env_id, level_str)