import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from collections import Counter

from TestingSolver import *

class DiffSolver(TestingSolver):
    def __init__(self, **params):
        super().__init__(**params)

    def predict(self, observation, info):
        action = np.ones(self.testing_env.action_space.shape)
        action /= action.shape[0]
        return action

    def train(self):
        pass

    def load(self):
        pass

    def save(self):
        pass
