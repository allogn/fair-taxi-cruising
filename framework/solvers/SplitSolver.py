import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from collections import Counter

from TestingSolver import *
from cA2CSolver import *

class SplitSolver(TestingSolver):
    def __init__(self, **params):
        super().__init__(**params)

    def predict(self, observation, info):
        # create a manager


        # for each idle driver - manager fisrt takes care of the correct distribution


        # manager sends closest idle drivers so to fix the distribution


        # withing each cell ask cA2C to solve the rest

        action = np.ones(self.testing_env.get_action_space_shape())
        action /= action.shape[0]
        return action

    def train(self, db_save_callback = None):
        # split a world into different worlds based on order logs

        # create cA2C per each world

        # train each cA2C using an estimate of average drivers, and adding drivers in random

        # make sure the model is saved

        pass

    def load(self):
        pass

    def save(self):
        pass
