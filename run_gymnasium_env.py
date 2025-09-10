import gymnasium
import gymnasium_env
from gymnasium.wrappers import FlattenObservation

env = gymnasium.make('gymnasium_env/GridWorld-v0')
wrapped_env = FlattenObservation(env)
print(wrapped_env.reset())     # E.g.  [3 0 3 3], {}