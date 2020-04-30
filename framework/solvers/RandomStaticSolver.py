import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from collections import Counter
import numpy as np
import networkx as nx

from TestingSolver import *


class RandomStaticSolver(TestingSolver):

    def __init__(self, **params):
        super().__init__(**params)

    def train(self, db_save_callback = None):
        world = nx.read_gpickle(os.path.join(self.dpath, "world.pkl"))
        self.A = self.get_random_A(world)

    @staticmethod
    def get_random_A(world):
        N = len(world)
        A = np.zeros((N,N))
        for i in range(N):
            A[i][i] = np.random.random()
            for j in world.neighbors(i):
                A[i][j] = np.random.random()
        A = (A.T / np.sum(A,axis=1)).T
        return A

    @staticmethod
    def sample_actions_based_on_A(A, world, context):
        actions = []
        for node in world.nodes():
            neighbors = list(world.neighbors(node)) + [node]
            idle_cars = context[0][node]
            targets = Counter(np.random.choice(neighbors, int(idle_cars), p=[A[node, l] for l in neighbors]))
            for k in targets:
                if node == k:
                    continue
                actions.append((node, k, targets[k]))
        return actions

    def get_dispatch_action(self, env, state, context):
        return self.sample_actions_based_on_A(self.A, env.world, context)

    def getA(self):
        return self.A

    def load(self):
        pass # refactor eventually

    def save(self):
        pass # refactor eventually

    def predict(self, state, info):
        action = np.ones(self.env.get_action_space_shape())
        action /= action.shape[0]
        return action
