import os, sys

from framework.solvers.GymSolver import GymSolver
from framework.solvers.cA2CSolver import cA2CSolver
from framework.solvers.cA2C.cA2C import *

from copy import deepcopy
from random import randint
import numpy as np
from collections import namedtuple
from tqdm import tqdm
import time
import logging

class RobustGymSolver(cA2CSolver):
    def __init__(self, **params):
        super().__init__(**params)

    def get_solver_signature(self):
        return "Robust" + super().get_solver_signature()

    def find_c(self, training_iteration):
        best_robust_reward, robust_threshold = -np.inf, 0
        gamma = self.params["robust_gamma"]
        cmin = self.params["cmin"]
        cmax = self.params["cmax"]
        epsilon = self.params["robust_epsilon"] # absolute reward error
        nu = self.params["robust_nu"]

        c = (cmax + cmin) / 2
        steps_log = []
        while abs((cmax + cmin)/2 - cmin) > epsilon:
            self.test_env.set_income_bound(c)
            stats = self.run_test_episode(training_iteration, draw=False, debug=False) 
            reward = np.sum(stats['rewards'])
            if self.DEBUG:
                logging.info("Trying c={}. Reward {}".format(c, reward))
            best_robust_reward = max(best_robust_reward, reward)

            robust_threshold = c * self.test_env.n_drivers * (1 - nu)
            possible = reward > c * self.test_env.n_drivers * (1 - nu)
            steps_log.append((c, reward - c * self.test_env.n_drivers * (1 - nu)))
            if possible:
                cmin = cmin + (c - cmin) * gamma
                c = (cmax + cmin) / 2
                if self.DEBUG:
                    logging.info("Possible. reward {}, thsh {}, gap {}".format(reward, robust_threshold, steps_log[-1][1]))
            else:
                cmax = cmax - (cmax - c) * gamma
                c = (cmax + cmin) / 2
                if self.DEBUG:
                    logging.info("Impossible. reward {}, thsh {}, gap {}".format(reward, robust_threshold, steps_log[-1][1]))

            if self.DEBUG:
                logging.info("Finished iteration with cmin, cmax, delta {}".format((cmin, cmax, cmax - cmin)))

        if self.DEBUG:
            logging.info("Finishing with final c {}".format(c))
        steps_log = sorted(steps_log)

        if self.DEBUG:
            for x in steps_log:
                print("{:10.4f}:{}".format(x[0], 1 if x[1] > 0 else 0))

        return c


    def train(self, db_save_callback = None):
        t1 = time.time()
        replay = ReplayMemory(memory_size=1e+6, batch_size=self.params['batch_size'])
        policy_replay = policyReplayMemory(memory_size=1e+6, batch_size=self.params['batch_size'])

        income_bounds = []
        global_step1 = 0
        global_step2 = 0

        self.log['time_batch'] = 0
        self.log['time_rollout'] = 0
        self.log['time_tests'] = 0

        # do preliminary test run
        self.run_tests(0, draw=self.params['draw'] == 1, verbose=0)

        if self.verbose:
            pbar = tqdm(total=self.params["iterations"], desc="Training cA2C (iters)")

        for n_iter in np.arange(self.params["iterations"]):
            seed = self.params['seed'] + n_iter + 123
            global_step1, global_step2 = self.do_iteration(n_iter, replay, policy_replay, seed, db_save_callback, pbar,
                                                            global_step1, global_step2)
            c = self.find_c(n_iter)
            income_bounds.append(c)
            self.env.set_income_bound(c)

        if self.verbose:
            pbar.close()

        self.log['train_time'] = time.time() - t1
        self.log['income_bounds'] = [float(c) for c in income_bounds]

        
