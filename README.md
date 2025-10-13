# COMP 4010 - Final Project

### Team Members: 
- Jackson Silver
- Kinjal Kamboj      - 101227444
- Alexis Udechukwu   - 101225811
- Trista Wang        - 
- Lujain Sharafeldin - 101246804



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