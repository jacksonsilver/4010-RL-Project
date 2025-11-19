import numpy as np

class ReplayBuffer:
    def __init__(self, step_size, gamma, max_step):
        self.step_size = step_size
        self.gamma = gamma
        self.max_step = max_step
        self.experiences: dict = {}

    def UpdateExperiences(self, state, action, reward, next_state):
        # Update Model with observed transition
        if state not in self.experiences:
            self.experiences[state] = {} # Map state to empty actions dictionary
        
        self.experiences[state][action] = (reward, next_state) # State + Action leads to reward, next_state

    def UpdateQ(self, Q):
        for _ in range(self.max_step):
            # 1. Randomly select previously observed state-action pair (s,a)
            s = np.random.choice(list(self.experiences.keys()))
            a = np.random.choice(list(self.experiences[s].keys()))

            # 2. Simulate taking action a in state s to get r, s'
            reward, next_state = self.experiences[s][a]

            # 3. Update Q(s,a) using simulated r, s'
            next_action = np.argmax(Q[next_state])
            Q[s, a] = Q[s, a] + self.step_size * (reward + self.gamma * Q[next_state, next_action] - Q[s, a])