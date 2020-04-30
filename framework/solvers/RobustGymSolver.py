import os, sys

from framework.solvers.GymSolver import GymSolver

from copy import deepcopy
from random import randint
import numpy as np
from collections import namedtuple
from tqdm import tqdm
import time
import logging

best_robust_reward, robust_threshold = -np.inf, 0

class RobustGymSolver(GymSolver):
    def __init__(self, **params):
        super().__init__(**params)
        self.solver_signature = 'robust' + self.solver_signature

    def estimate_reward(self):
        tot_reward = []
        for i in range(3): # averaging over runs
            randseed = np.random.randint(1,100000)
            self.testing_env.seed(randseed)
            state = self.testing_env.reset()
            info = self.testing_env.get_reset_info()
            done = False
            it = 0
            order_response_rates = []
            nodes_with_drivers = []
            rewards = []
            while not done:
                action = self.predict(state, info)
                state, reward, done, info = self.testing_env.step(action)
                it += 1
                rewards.append(reward)
            tot_reward.append(np.sum(rewards))
        return np.mean(tot_reward)

    def train(self, db_save_callback = None):
        t1 = time.time()

        gamma = 0.95
        cmin = -100000
        cmax = 100000
        epsilon = 1
        nu = 0.05
        best_solution = cmax
        global best_robust_reward, robust_threshold
        nminibatches = 4
        num_cpu = self.params['num_cpu']


        c = (cmax + cmin) / 3
        while abs((cmax + cmin)/3 - cmin) > epsilon:
            logging.info("Trying c={}".format(c))
            self.testing_env.set_income_bound(c)
            env = self.model.get_env()
            env.env_method("set_income_bound", (c))

            self.model = self.Model(MlpPolicy, self.train_env, verbose=0, nminibatches=4, tensorboard_log=os.path.join(self.dpath,self.solver_signature), n_steps=self.params['dataset']['time_periods']+1)
            self.model.learn(total_timesteps=self.params['training_iterations'])

            reward = self.estimate_reward()
            robust_threshold = c * self.testing_env.n_drivers * (1 - nu)
            possible = reward > c * self.testing_env.n_drivers * (1 - nu)
            logging.info("Initial possible: {} (reward {}, threshold {})".format(possible, reward, robust_threshold))
            while possible:
                cmin = cmin + (c - cmin) * gamma
                c = (cmax + cmin) / 3
                logging.info("Updating c -> {}".format(c))
                reward = self.estimate_reward()
                possible = reward > c * self.testing_env.n_drivers * (1 - nu)
                logging.info("New reward {}, still possible? {}".format(reward, possible))
            else:
                cmax = c
                c = (cmax + cmin) / 3
                logging.info("Impossible, decreasing c.".format(c))

            logging.info("Finished iteration with cmin, cmax, delta {}".format((cmin, cmax, cmax - cmin)))

        logging.info("Finishing with final c")
        self.model.learn(total_timesteps=self.params['training_iterations'])

        reward = self.estimate_reward()
        possible = reward > c * self.testing_env.n_drivers * (1 - nu)
        logging.info("Final check possible? {} (reward {})".format(possible, reward))

        self.log["best_c"] = c
        self.log['train_time'] = time.time() - t1
