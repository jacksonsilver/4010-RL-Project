# COMP 4010 - Final Project

### Team Members: 
| Name               | Student ID |
|------------------- |------------|
| 1. Jackson Silver     | 101224148  |
| 2. Kinjal Kamboj      | 101227444  | 
| 3. Alexis Udechukwu   | 101225811  |
| 4. Trista Wang        | 101231212  |
| 5. Lujain Sharafeldin | 101246804  |

### MDP Explanation

4 Actions: Left, Down, Right, Up

States are represented in training as numbers, but in the environment they are represented as:
    (X, Y, W-MASK, AVAIL-ACTION-MASK)

Where X and Y are the tile's position

W-MASK is a bit mask indicating if it's a water tile (1) or not
And Avail-Action-Mask is a bit mask indicating what actions are available at that position.

So, 1111 means all actions are available,
1010, means that the agent can go left and right, but not up or down.
  
## Installation

To install your new environment, run the following commands:

1. Set up an environment

```{shell}
python -m venv 4010Project
.\4010Project\Scripts\activate
python -m pip install -U pip #just to update
```

2. Install Requirements in project directory
```(shell)
pip install -r requirements.txt
pip install -e .
```

## Running for Testing
- Run  `gymnasium_env/envs/v0_thin_ice_env.py` 

## Running Algorithms (QLearning & PPO)
- Run  `gymnasium_env\envs\main.py` 


