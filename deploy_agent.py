import os
import sys
import argparse
from typing import Final

from gymnasium_env.envs.v1_thin_ice_train import ThinIceDQNAgent

PATH_TO_LEVELS: Final[str] = './level_txt_files/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deploy a trained Thin Ice DQN agent.")
    parser.add_argument('level', type=str)
    args = parser.parse_args()
    level_to_deploy = args.level
    print(f"--- Deploying Agent on: {level_to_deploy} ---")

    project_root = os.path.dirname(os.path.abspath(__file__))
    _LEVEL_DIR = os.path.join(project_root, 'level_txt_files')
    level_path = os.path.join(_LEVEL_DIR, level_to_deploy)

    # --- Load the agent definition ---
    all_levels_dummy = [f for f in os.listdir(_LEVEL_DIR) if f.endswith('.txt')] if os.path.isdir(_LEVEL_DIR) else []

    # Need level_list just to construct the correct agent save name
    agent_multi = ThinIceDQNAgent('thin-ice-v1', level_list=all_levels_dummy)
    agent_multi.level_str = level_to_deploy # Set the actual level for deployment

    # --- Pass custom objects when calling deploy ---
    # The deploy method inside the agent handles loading with custom objects now
    agent_multi.deploy(render=True)
