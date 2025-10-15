# COMP 4010 - Final Project

### Team Members: 
- Jackson Silver     - 101224148
- Kinjal Kamboj      - 101227444
- Alexis Udechukwu   - 101225811
- Trista Wang        - 
- Lujain Sharafeldin - 101246804

### Update 15th October
What has been done past two weeks:
- Everyone set up the environment
- Designed the interface from scratch along with the assets used (tiles, players, target, walls)
- Started with console terminal output to display actions taken, transitioned to working on the interface to display assets with pygame
- Started training model with different levels (mazes) to see if agent is able to learn (currently interface shows agent moving to target successfully after finishing training)

What will be done the next two weeks:
 
  
## Installation

To install your new environment, run the following commands:

1. Set up an environment (optional but recommended)

```{shell}
python -m venv 4010Project
.\4010Project\Scripts\activate
python -m pip install -U pip #just to update
```
**Note**: Do not push the 4010Project environment folder to GitHub.

2. Install Requirements in project directory
```(shell)
pip install -r requirements.txt
pip install -e .
```

## Running for Testing
- Run  `gymnasium_env/envs/v0_thin_ice_env.py` 
- Run  `gymnasium_env/envs/v0_thin_ice_train.py` 



##
# Gymnasium Examples
Some simple examples of Gymnasium environments and wrappers.
For some explanations of these examples, see the [Gymnasium documentation](https://gymnasium.farama.org).

### Environments
This repository hosts the examples that are shown [on the environment creation documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).
- `GridWorldEnv`: Simplistic implementation of gridworld environment

### Wrappers
This repository hosts the examples that are shown [on wrapper documentation](https://gymnasium.farama.org/api/wrappers/).
- `ClipReward`: A `RewardWrapper` that clips immediate rewards to a valid range
- `DiscreteActions`: An `ActionWrapper` that restricts the action space to a finite subset
- `RelativePosition`: An `ObservationWrapper` that computes the relative position between an agent and a target
- `ReacherRewardWrapper`: Allow us to weight the reward terms for the reacher environment

### Contributing
If you would like to contribute, follow these steps:
- Fork this repository
- Clone your fork
- Set up pre-commit via `pre-commit install`

PRs may require accompanying PRs in [the documentation repo](https://github.com/Farama-Foundation/Gymnasium/tree/main/docs).


## Installation

```{shell}
cd gymnasium_env
pip install -e .
```
