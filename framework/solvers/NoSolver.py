from framework.solvers.TestingSolver import *
import numpy as np

class NoSolver(TestingSolver):
    def __init__(self, **params):
        super().__init__(**params)

    def train(self):
        # do testing instead of training, so that we have some stats to compare right away
        self.run_tests()

    def test(self):
        pass

    def load(self):
        pass

    def save(self):
        pass

    def predict(self, observation, info):
        action = np.zeros(self.testing_env.get_action_space_shape())
        wl = len(self.testing_env.world)
        last_ind = action.shape[0] // wl
        for i in range(wl):
            action[i*last_ind-1] = 1
        return action
