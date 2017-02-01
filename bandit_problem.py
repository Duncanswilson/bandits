import numpy as np
import random

class Bandit(object):
    
    def __init__(self, mu = 0.0, sigma = 1.0):
        self.mu = mu
        self.sigma = sigma

    def pull(self):
        return self.sigma * np.random.randn() + self.mu

class MultiArmBandit(object):

    def __init__(self, n, params = []):
        if n < 1:
            raise ValueError("Number of bandits must be positive.")
        self.n = n
        if params == []:
            self.bandits = [Bandit(random.randint(0, 20), 
                                   random.randint(0, 5)) for i in range(n)]
        else:
            if n != len(params):
                raise ValueError("Invalid number of params.")
            self.bandits = []
            for i in range(n):
                self.bandits.append(Bandit(params[i][0], params[i][1]))

    def pull(self):
        return [b.pull() for b in self.bandits]

class BanditProblem(object):

    def __init__(self, sz, params = []):
        self.n = sz
        self.mab = MultiArmBandit(sz, params)

        self.action_space = range(sz)

    def step(self, action):
        rewards = self.mab.pull()
        return rewards[action]

