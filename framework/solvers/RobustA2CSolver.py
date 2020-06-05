import os, sys

from framework.solvers.cA2CSolver import cA2CSolver
from framework.solvers.cA2C.cA2C import *

from copy import deepcopy
from random import randint
import numpy as np
from collections import namedtuple
from tqdm import tqdm
import time
import logging

class RobustA2CSolver(cA2CSolver):
    def __init__(self, **params):
        super().__init__(**params)

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
            stats = self.run_test_episode(training_iteration, draw=False, debug=self.DEBUG) 
            reward = np.sum(stats['driver_income_bounded'])
            best_robust_reward = max(best_robust_reward, reward)

            robust_threshold = c * self.test_env.n_drivers * (1 - nu)
            possible = reward > c * self.test_env.n_drivers * (1 - nu)
            steps_log.append((c, reward, c * self.test_env.n_drivers, reward - c * self.test_env.n_drivers * (1 - nu)))
            if possible:
                cmin = cmin + (c - cmin) * gamma
                c = (cmax + cmin) / 2
            else:
                cmax = cmax - (cmax - c) * gamma
                c = (cmax + cmin) / 2
            
        logging.info("Finishing with final c={}".format(c))
        steps_log = sorted(steps_log)
        self.log['step_log_{}'.format(training_iteration)] = np.array(steps_log, dtype=float).tolist()

        if self.DEBUG:
            for x in steps_log:
                logging.info("c={:10.4f}:rew={}:gap={}".format(x[0], x[1], x[3]))

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

        
