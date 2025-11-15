class DecayingEpsilon:
    def __init__(self, start_epsilon: float, end_epsilon: float, decay_rate: float):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_rate = decay_rate
        self.epsilon = start_epsilon
    
    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon
    
    def get_epsilon(self) -> float:
        return self.epsilon

    def update(self, episode: int):
        y = (episode ** 2) / (self.decay_rate + episode)
        self.epsilon = max(self.end_epsilon, self.start_epsilon / (1 + y))