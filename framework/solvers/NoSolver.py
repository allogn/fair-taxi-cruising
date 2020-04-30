from framework.solvers.TestingSolver import *
import numpy as np

class NoSolver(TestingSolver):
    def __init__(self, **params):
        super().__init__(**params)

    def train(self, db_save_callback = None):
        # do testing instead of training, so that we have some stats to compare right away
        self.run_tests(0, draw = True)

    def test(self):
        pass

    def load(self):
        pass

    def save(self):
        pass

    def predict(self, observation, info):
        action = np.zeros(self.test_env.get_action_space_shape())
        wl = len(self.test_env.world)
        last_ind = action.shape[0] // wl
        for i in range(wl):
            action[i*last_ind-1] = 1
        return action
