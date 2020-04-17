import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from collections import Counter
import numpy as np
from framework.solvers.TestingSolver import TestingSolver

class DiffSolver(TestingSolver):
    def __init__(self, **params):
        super().__init__(**params)

    def predict(self, observation, info):
        action = np.ones(self.testing_env.get_action_space_shape())
        action /= action.shape[0]
        return action

    def test(self):
        pass

    def train(self):
        self.run_tests()

    def load(self):
        pass

    def save(self):
        pass
