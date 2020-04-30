import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from TestingSolver import *
from ValIterSolver import *

from copy import deepcopy
from random import randint
import numpy as np
from collections import namedtuple
from tqdm import tqdm
import time


class RobustValIterSolver(TestingSolver):
    def __init__(self, **params):
        super().__init__(**params)

        self.vsolver = ValIterSolver(**self.params)

    def get_dispatch_action(self, env, state, context):
        return self.vsolver.get_dispatch_action(env, state, context)

    def train(self, db_save_callback = None):
        t1 = time.time()

        gamma = 0.9
        cmin = -100000
        cmax = 100000
        epsilon = 0.01
        nu = 0.1
        best_solution = None

        while abs((cmax + cmin)/2 - cmin) > epsilon:
            c = (cmax + cmin) / 2
            vsolver = ValIterSolver(**self.params)
            vsolver.train(maxincome = c)
            total_revenue = vsolver.total_reward
            possible = total_revenue > c * vsolver.env.n_drivers * (1 - nu)

            if possible:
                cmin = cmin + (c - cmin) * gamma
                best_solution = vsolver.dyn_value_table
            else:
                cmax = c

            print(possible, cmin, cmax, cmax - cmin)

        self.log['train_time'] = time.time() - t1
        assert(best_solution is not None)
        pkl.dump(best_solution, open(os.path.join(self.dpath, "trained_ValIter.pkl"), "wb"))

    def load(self):
        pass # refactor eventually

    def save(self):
        pass # refactor eventually
