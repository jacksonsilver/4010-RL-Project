import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys
from collections import deque
import multiprocessing
from typing import Dict, Final, Type, Any, Optional, Tuple 

import torch
import torch.nn as nn
from stable_baselines3 import DQN

from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.policies import BasePolicy

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.noise import ActionNoise 

from gymnasium_env.envs.thin_ice_training_agent import ThinIceTrainingAgent

PATH_TO_LEVELS: Final[str] = './level_txt_files/'

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        obs_space = observation_space['obs']
        n_input_channels = obs_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(obs_space.sample()[None]).float()
                ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs_tensor_float = observations['obs'].float()
        return self.linear(self.cnn(obs_tensor_float))

class MaskableDQNPolicy(DQNPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch=None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CustomCNN,
        features_extractor_kwargs: Dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] | None = None,
    ):
        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        
        if self.features_extractor_kwargs is None:
            self.features_extractor_kwargs = {}
        self.features_extractor = self.features_extractor_class(
            self.observation_space, **self.features_extractor_kwargs
        )
        self.features_dim = self.features_extractor.features_dim

        if net_arch is None:
            net_arch = [64, 64]
        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "features_extractor": self.features_extractor,
            "features_dim": self.features_dim,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": self.normalize_images,
        }

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_q_net(self) -> QNetwork:
        return QNetwork(**self.net_args).to(self.device)

    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        features = self.features_extractor(observation)

        q_values = self.q_net.q_net(features)

        action_mask = observation['action_mask'].bool().to(q_values.device)
        masked_q_values = torch.where(action_mask, q_values, torch.tensor(-1e8, dtype=torch.float32, device=q_values.device))

        action = torch.argmax(masked_q_values, dim=1)
        return action.reshape(-1)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
         features = self.features_extractor(obs)
         q_values = self.q_net.q_net(features)
         return q_values
    
class MaskableDQN(DQN):

    # Overriding sample action function so agent will only take valid moves
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: None,
        num_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:

        if self.num_timesteps < learning_starts:
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(self._last_obs, deterministic=True)

            return unscaled_action, unscaled_action

        # --- Epsilon-greedy exploration ---
        if np.random.rand() < self.exploration_rate:
            action_masks = self._last_obs["action_mask"]
            unscaled_action = np.zeros(num_envs, dtype=self.action_space.dtype)

            for i in range(num_envs):
                env_mask = action_masks[i]
                valid_actions = np.where(env_mask == 1)[0]
                if len(valid_actions) > 0:
                    unscaled_action[i] = np.random.choice(valid_actions)
                else:
                    unscaled_action[i] = 0 
        else:
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        return unscaled_action, unscaled_action

class ThinIceDQNAgent(ThinIceTrainingAgent):
    def __init__(self, env_id: str ='thin-ice-v1', level_str: str = 'level_0.txt', level_list: list[str] = None):
        super().__init__('DQN', env_id, level_str)
        self.env_id = env_id
        self.level_list = level_list
        self.level_str = level_str

        if self.level_list and len(self.level_list) > 0:
            self.reference_name = f"{self.env_id}-DQN-MultiLevel-Masked"

        else:
            self.reference_name = f"{self.env_id}-DQN-{self.level_str.split('.')[0]}-Masked"
        self.model_path = os.path.join(self.getPkFolderPath('DQN'), self.reference_name + '_model.zip')

    def train(self, total_timesteps: int = 100_000, start_learning_rate: float = 1e-4, end_learning_rate: float = 1e-5, gamma: float = 0.99, num_cpu: int = 4, **kwargs):

        print(f"--- Starting/Resuming DQN Training for: {self.reference_name} (with Action Masking) ---")
        print(f"Target Total Timesteps for this run: {total_timesteps}")
        print(f"Using {num_cpu} parallel environments.")
        print(f"Learning Rate Schedule: Linear from {start_learning_rate} to {end_learning_rate}")

        env_kwargs = {'max_rows': 15, 'max_cols': 19}

        # Check for training with level list or just one level
        if self.level_list: 
            env_kwargs['level_list'] = self.level_list 
            print(f"Training on {len(self.level_list)} levels.")

        else: 
            env_kwargs['level_str'] = self.level_str
            print(f"Training on single level: {self.level_str}")

        env = make_vec_env(self.env_id, n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv, env_kwargs=env_kwargs)

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=128) 
        )

        lr_schedule = get_linear_fn(start=start_learning_rate, end=end_learning_rate, end_fraction=1.0)

        reset_learn_timesteps = True

        custom_objects = {
            "policy": MaskableDQNPolicy,
            "policy_kwargs": policy_kwargs,
        }

        # Check to continue training existing model
        if os.path.exists(self.model_path):
            print(f"Loading parameters from existing model: {self.model_path}")

            old_model = MaskableDQN.load(self.model_path, custom_objects=custom_objects)
            learned_params = old_model.policy.state_dict()
            del old_model

            print("Creating new model instance with fresh exploration schedule...")

            model = MaskableDQN(
                MaskableDQNPolicy, 
                env, 
                learning_rate=lr_schedule, 
                gamma=gamma,
                buffer_size=100_000, 
                batch_size=128, 
                learning_starts=10000,
                target_update_interval=1000, 
                exploration_fraction=0.4, 
                exploration_final_eps=0.01,
                tensorboard_log=f"./tensorboard_logs/{self.reference_name}/",
                verbose=1, policy_kwargs=policy_kwargs 
            )
            model.policy.load_state_dict(learned_params)
            print("Transferred learned weights to the new model.")

        else:
            print("No existing model found. Starting new training.")

            model = MaskableDQN(
                MaskableDQNPolicy, 
                env, 
                learning_rate=lr_schedule, 
                gamma=gamma,
                buffer_size=100_000, 
                batch_size=128, 
                learning_starts=10000,
                target_update_interval=1000, 
                exploration_fraction=0.4, 
                exploration_final_eps=0.01,

                tensorboard_log=f"./tensorboard_logs/{self.reference_name}/",
                verbose=1, policy_kwargs=policy_kwargs 
            )

        print("\n--- Model Training ---")
        model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=self.reference_name,
            reset_num_timesteps=reset_learn_timesteps
        )

        print(f"Training complete. Saving final model to: {self.model_path}")
        model.save(self.model_path)
        env.close()

    def deploy(self, render: bool = True, max_steps: int = 500):
        print(f"--- Deploying model {self.reference_name} on level {self.level_str} ---")

        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found at {self.model_path}")
            return

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=128)
        )

        custom_objects = {
            "policy": MaskableDQNPolicy,
            "policy_kwargs": policy_kwargs,
            "lr_schedule": get_linear_fn
        }

        model = MaskableDQN.load(self.model_path, custom_objects=custom_objects)

        env = gym.make(
            self.env_id, level_str=self.level_str, render_mode="human" if render else None,
            max_rows=15, max_cols=19
        )

        print("Loading level for deployment...")
        obs_dict, info = env.reset()

        if env.unwrapped._level:
            max_steps = max(max_steps, env.unwrapped._level.n_visitable_tiles * 2)
            print(f"Level loaded. Setting max_steps to {max_steps}")

        else:
            print("Warning: Could not determine level size after reset. Using default max_steps.")
            max_steps = max(max_steps, 200) 

        try:
            terminated = False
            step_count = 0
            total_reward = 0

            while (not terminated and step_count < max_steps):
                step_count += 1

                batched_obs_dict = {
                    key: np.expand_dims(value, axis=0)
                    for key, value in obs_dict.items()
                }

                action_batch, _states = model.predict(batched_obs_dict, deterministic=True)
                action = action_batch[0]

                obs_dict, reward, terminated, truncated, info = env.step(action)

                if render:
                    env.render()

                total_reward += reward
                if terminated or truncated:
                    print(f"Deployment finished after {step_count} steps.")
                    print(f"Final Reward: {total_reward}")

                    if render:
                        import time
                        try:
                            time.sleep(2) 
                        except KeyboardInterrupt:
                            print("Pause interrupted.")

        finally: 
            env.close()

if __name__ == '__main__':
    num_cpu_to_use = max(1, multiprocessing.cpu_count() - 1)
    print("--- Starting Multi-Level Agent Training ---")
    
    all_levels = [f for f in os.listdir(PATH_TO_LEVELS) if f.endswith('.txt')] 

    if all_levels:
        print(f"Found {len(all_levels)} levels: {all_levels}")
    else:
        print("No levels found. Exiting now")
        sys.exit()

    agent_multi = ThinIceDQNAgent('thin-ice-v1', level_list=all_levels)

    agent_multi.train(
        total_timesteps=5000000, 
        start_learning_rate=1e-4,  
        end_learning_rate=1e-5, 
        gamma=0.99,            
        num_cpu=num_cpu_to_use
    )

    print("--- Training complete ---")
    print(f"Model saved to: {agent_multi.model_path}")
    print("\nRun 'python deploy_agent.py [level_name.txt]' to watch.")