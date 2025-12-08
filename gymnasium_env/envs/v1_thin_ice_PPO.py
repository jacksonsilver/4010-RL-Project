import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import gymnasium as gym
import pygame
import math

from thin_ice_training_agent import ThinIceTrainingAgent
import v0_thin_ice_env as ti
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init(module, weight_init, bias_init, gain=1):
    """Helper to initialize a layer weight and bias."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Flatten(nn.Module):
    """Helper to flatten a tensor."""
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNPolicy(nn.Module):
    def __init__(self, action_size, num_channels=7, height=15, width=19):
        super(CNNPolicy, self).__init__()

        # Define activation functions
        self.relu = nn.ReLU()

        # CNN layers
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Dummy forward pass to compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, height, width)
            dummy = self.relu(F.max_pool2d(self.conv1(dummy), 2))
            dummy = self.relu(F.max_pool2d(self.conv2(dummy), 2))
            dummy = self.relu(F.max_pool2d(self.conv3(dummy), 2))
            self.flattened_size = dummy.view(1, -1).shape[1]

        # Fully connected layers
        self.fc_pi = nn.Linear(self.flattened_size, action_size)
        self.fc_v = nn.Linear(self.flattened_size, 1)



    def pi(self, x, action_mask=None):
        x = self.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)
        logits = self.fc_pi(x)

        if action_mask is not None:
            # action_mask: shape [batch_size, num_actions], dtype: bool
            masked_logits = logits.clone()
            #masked_logits[~action_mask] = float('-inf')
            masked_logits = masked_logits.masked_fill(~action_mask.squeeze(0), float('-inf'))

            probs = F.softmax(masked_logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)

        return probs

    
    def v(self, x):
        # Same CNN layers for value function
        x = self.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.relu(F.max_pool2d(self.conv3(x), 2))

        x = x.view(x.size(0), -1) 
        return self.fc_v(x) 

class ThinIcePPOAgent(ThinIceTrainingAgent):
    def __init__(self, env_id='thin-ice-v0', level_str='level_1.txt'):
        super().__init__("PPO",env_id, level_str)
        self.total_losses =[]
    
    def train(self, gamma=0.99, learning_rate=0.0003, n_episodes=2000):
        print("===== Beginning Training")
        env: ti.ThinIceEnv = gym.make(self.env_id, level_str=self.level_str)
        
        actions = env.unwrapped.n_actions
        states = env.unwrapped.n_states
        self.tile_types = tile_types = env.unwrapped.level.n_tile_types
        self.rows = rows = env.unwrapped.level.n_rows
        self.cols = cols = env.unwrapped.level.n_cols

        ### DEBUGGING PRINTS
        # print(f'START [DEBUG]')
        # print(f'states: {states}')
        # print(f'actions {actions}')
        # print(f'tile types {self.tile_types}')
        # print(f'number of rows {self.rows}')
        # print(f'number of cols {self.cols}')
        # print(f'END [DEBUG]')

        #PPO Parameters
        self.gamma = gamma
        self.lmbda = 0.95
        self.eps_clip = 0.2
        self.k_epoch = 10
        self.v_coef = 0.5

        self.entropy_start = 0.5
        self.entropy_end   = 0.01
        self.decay_rate = 0.001



        #Initializing CNN (neural network of all states)
        self.policy_network = CNNPolicy(actions,tile_types+2,rows,cols).to(device)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        #rollout buffer -> keep track of everything
        self.memory = {
            'state': [], 'action': [], 'reward': [], 'next_state': [],
            'action_prob': [], 'terminated': [], 'count': 0,
            'advantage': [], 'td_target': torch.FloatTensor([]),
            'mask':[]
        }

        prev_len = 0
        episode_rewards = []

        #training loop
        for i in range(n_episodes):
            print(f'============== EPISODE {i} ==============')
            self.entropy_coef = self.entropy_end + (self.entropy_start - self.entropy_end) * math.exp(-self.decay_rate * i)


            state = env.reset()[0]
            terminated = False
            episode_reward = 0
            

            #Each epside
            while not terminated:
                state_tensor = self.state_index_to_tensor(env,state)

                raw_mask = env.unwrapped.get_available_actions_mask(state)
                available_actions = env.unwrapped.action_mask_to_actions(raw_mask)
                actions_bool = env.unwrapped.get_actions_boolean_list(available_actions)

                # print(f"AGENT CAN ONLY GO: {raw_mask}")
                # print(f"WHICH MEANS AGENT CAN ONLY GO {available_actions}")
                # print(f"boolean looks like: {actions_bool}")

                action_mask = torch.tensor(actions_bool, dtype=torch.bool).unsqueeze(0).to(device)

                # print(f'my avail mask is: {action_mask}')
                # print("test")

                
                #action probabilities
                with torch.no_grad():
                    probs = self.policy_network.pi(state_tensor,action_mask)
                    print(f"Action mask: {action_mask.cpu().numpy()}")
                    print(f"Probabilities: {probs.cpu().numpy()}")
    
                    value = self.policy_network.v(state_tensor)

                #if agent dies there's a whole thing    
                if not action_mask.any():
                    print("Agent died! no valid actions")
                    
                    terminated = True
                    episode_reward -=1 
                    env.render()
                    pygame.quit()
                    #self.add_to_memory(state_tensor, torch.tensor([0]), reward, state_tensor, 0.0, terminated, action_mask)
                    break
                
                else:

                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    action_prob =probs[0,action.item()].item()
                    print(f' this is action.item whtever: {action.item()}')

                    next_state,reward,terminated,_,info = env.step(action.item())
                    next_state_tensor = self.state_index_to_tensor(env,next_state)
                    episode_reward += reward

                    if info['target_reached'] and info['all_tiles_covered']:
                        self.entropy_start =0.0
                        self.entropy_end  = 0.0
                        self.decay_rate = 0.0
                        print('task completed!')
                        self.entropy_coef = 0.0 # no more exploring, this is the best we;ve got
                                #save policy
                        path = self.getPkFolderPath('PPO')
                        filename = self.reference_name + '_solution-WORKING.pt'
                        torch.save(self.policy_network.state_dict(),os.path.join(path,filename))
                        print(f'Saved model to {filename}')
                        self.plot_graph(episode_rewards)
                        env.close()
                        break


                    #store values
                    self.add_to_memory(state_tensor,action,reward,next_state_tensor,action_prob,terminated,action_mask)

                    #updating states
                    state_tensor = next_state_tensor
                    state = next_state

            #Calculate target and advantage
            length = len(self.memory['state']) - prev_len
            self.compute_target_and_advantages(length)
            prev_len = len(self.memory['state'])

            #update every 10 epsiodes
            if (i+1)% self.k_epoch == 0:
                    self.update_network(length)
                    prev_len = 0

            episode_rewards.append(episode_reward)

            print(f"Episode {i} reward: {episode_reward}")           
        
        #save policy
        path = self.getPkFolderPath('PPO')
        filename = self.reference_name + '_solution.pt'
        torch.save(self.policy_network.state_dict(),os.path.join(path,filename))
        print(f'Saved model to {filename}')
        self.plot_graph(episode_rewards)
        env.close()
        return
        
    def state_index_to_tensor(self,env,state_index):
        cell = env.unwrapped.to_cell[state_index]
        x, y = cell[0], cell[1]

        
        # print(f'START [DEBUG]')
        # print(f' x type: {x}')
        # print(f'tile types {self.tile_types}')
        # print(f'number of rows {self.rows}')
        # print(f'number of cols {self.cols}')
        # print(f'END [DEBUG]')

        tensor = torch.zeros(self.tile_types+2,self.rows,self.cols)


        #fill in values
        for j in range(self.rows):
            for i in range(self.cols):
                tile = env.unwrapped.level.get_tile((i,j))
                tile_type = tile.tile_type.value
                tensor[tile_type,j,i] = 1.0

        visited_channel = self.tile_types  # the extra channel index
        print(env.unwrapped.visited_tiles)
        for tile in env.unwrapped.visited_tiles:
            vx, vy = tile[0], tile[1]
            tensor[visited_channel, vy, vx] = 1.0
        
        #player position in last channel
        player_channel = self.tile_types + 1
        tensor[player_channel, y, x] = 1.0

        return tensor.unsqueeze(0) #shape [1,C,H,W]
    
    def add_to_memory(self,s,a,r,next_s,prob,terminated,mask):

        #s & next_s are tensors
        self.memory['state'].append(s.squeeze(0))  
        self.memory['action'].append(a)
        self.memory['reward'].append(r)
        self.memory['next_state'].append(next_s.squeeze(0))
        self.memory['action_prob'].append(prob)
        self.memory['terminated'].append(terminated)
        self.memory['mask'].append(mask.squeeze(0))

    def compute_target_and_advantages(self,length):
        states = torch.stack(self.memory['state'][-length:]).to(device)
        next_states = torch.stack(self.memory['next_state'][-length:]).to(device)
        rewards = torch.tensor(self.memory['reward'][-length:], dtype=torch.float32).to(device)
        terminals = torch.tensor(self.memory['terminated'][-length:], dtype=torch.float32).to(device)

        #td_target
        with torch.no_grad():
            values = self.policy_network.v(states).view(-1)
            next_values = self.policy_network.v(next_states).view(-1)

        # TD target: r + γV(s') * (1 - done)
        td_target = rewards + self.gamma * next_values * (1 - terminals)

        #GAE advantage
        advantages = torch.zeros_like(rewards)
        gae = 0
        # print(f"values shape: {values.shape}")
        # print(f"next_values shape: {next_values.shape}")

        for t in reversed(range(len(rewards))):

            delta = rewards[t] + self.gamma * next_values[t] * (1 - terminals[t]) - values[t]
            #print(f"GAE shape at t={t}: {gae.shape}")
            gae = delta + self.gamma * self.lmbda * (1 - terminals[t]) * gae
            advantages[t] = gae

       # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.memory['advantage'] = advantages
        self.memory['td_target'] = td_target.unsqueeze(1)
    
    def update_network(self, length):
        if len(self.memory['state']) == 0:
            return

        start = len(self.memory['state']) - length
        end   = len(self.memory['state'])

        # Windowed tensors
        states_tensor  = torch.stack(self.memory['state'][start:end]).to(device)         # [L, C, H, W]
        actions_tensor = torch.stack(self.memory['action'][start:end]).long().to(device).view(-1,1)  # [L,1]
        old_probs      = torch.tensor(self.memory['action_prob'][start:end], dtype=torch.float32, device=device).view(-1)  # [L]
        masks_tensor =   torch.stack(self.memory['mask'][start:end]).to(device)# [L, A]


        # These were already saved for just the last `length` steps — don't slice again
        advantages = torch.as_tensor(self.memory['advantage'], dtype=torch.float32, device=device).view(-1)  # [L]
        td_target  = self.memory['td_target'].to(device).view(-1)                                           # [L]

        for _ in range(self.k_epoch):
            pi   = self.policy_network.pi(states_tensor,masks_tensor)                 # [L, A]
            newp = torch.gather(pi, 1, actions_tensor).squeeze(1)        # [L]
            ratio = newp / (old_probs + 1e-8)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            v_pred = self.policy_network.v(states_tensor).squeeze(1)     # [L]
            value_loss = 0.5 * (v_pred - td_target).pow(2).mean()

            entropy = torch.distributions.Categorical(pi).entropy().mean()

            loss = policy_loss + self.v_coef * value_loss - self.entropy_coef * entropy
            self.total_losses.append(loss.item())
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            if hasattr(self, 'scheduler'): self.scheduler.step()

        # Clear the right keys
        for k in ['state','action','reward','next_state','terminated','action_prob','advantage','mask']:
            self.memory[k] = []
        self.memory['td_target'] = torch.FloatTensor([]).to(device)

    def exponential_moving_average(self,data, alpha=0.05):
        ema = []
        for i, reward in enumerate(data):
            if i == 0:
                ema.append(reward)
            else:
                ema.append(alpha * reward + (1 - alpha) * ema[-1])
        return ema

    def plot_graph(self,rewards,):
        ema_rewards = self.exponential_moving_average(rewards, alpha=0.05)

        plt.figure(figsize=(10, 5))

        plt.plot(range(len(rewards)), rewards, alpha=0.3, label='Raw Reward')
        plt.plot(range(len(ema_rewards)), ema_rewards, color='blue', linewidth=2, label='Smoothed Reward (EMA)')
       
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Agent Learning Curve')
        plt.grid(True)
        plt.legend()

        path = self.getGraphFolderPath('PPO')
        filename = self.reference_name + '-reward-graph.png'
        plt.savefig(os.path.join(path, filename))
        print(f'Generated Graph: {filename}')

        # ----- LOSS CURVE -----
        if len(self.total_losses) > 0:
            plt.figure(figsize=(10,5))
            plt.plot(self.total_losses, color='red', linewidth=2, label='Total PPO Loss')
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.title("PPO Loss Curve")
            plt.grid(True)
            plt.legend()

            filename = self.reference_name + "-loss-graph.png"
            plt.savefig(os.path.join(path, filename))
            print(f"Generated Loss Graph: {filename}")



    def deploy(self, render=True, max_steps=500):
        print("====== Deployment")
        env: ti.ThinIceEnv = gym.make(self.env_id, level_str=self.level_str,render_mode = 'human')

        actions = env.unwrapped.n_actions
        self.tile_types = tile_types = env.unwrapped.level.n_tile_types
        self.rows = rows = env.unwrapped.level.n_rows
        self.cols = cols = env.unwrapped.level.n_cols

        # Load trained policy
        self.policy_network = CNNPolicy(actions, tile_types+2, rows, cols).to(device)
        path = self.getPkFolderPath('PPO')
        filename = self.reference_name + '_solution.pt'
        model_path = os.path.join(path,filename)

        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        self.policy_network.load_state_dict(state_dict)
        self.policy_network.eval()

        state = env.reset()[0]
        terminated = False
        steps = 0
        total_reward = 0
        
        while not terminated and steps < max_steps:


            #get action    
            state_tensor = self.state_index_to_tensor(env, state)
            raw_mask = env.unwrapped.get_available_actions_mask(state)
            available_actions = env.unwrapped.action_mask_to_actions(raw_mask)
            actions_bool = env.unwrapped.get_actions_boolean_list(available_actions)
            action_mask = torch.tensor(actions_bool, dtype=torch.bool).unsqueeze(0).to(device)
            print(action_mask)

            with torch.no_grad():
                probs = self.policy_network.pi(state_tensor, action_mask)   
                action = torch.argmax(probs, dim=-1)
            
            if not action_mask.any(): 
                terminated = True
                break

            next_state, reward, terminated, _, _ = env.step(action.item())
            total_reward += reward
            state = next_state
            steps += 1

        if render and terminated:
            filename = self.reference_name + f'_agent_final_path.png'
            filepath = self.getAgentSolutionsFolderPath("PPO")
            pygame.image.save(pygame.display.get_surface(), os.path.join(filepath, filename))

            print(f'Created Snapshot of final path: {filename}')
            #pygame.time.wait(1000) 

        print(f"Deployment finished. Total reward: {total_reward}")
        env.close()
        pygame.quit()

        return