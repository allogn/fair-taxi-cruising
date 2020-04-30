from framework.solvers.TestingSolver import *
from framework.FileManager import *

import numpy as np
import pickle as pkl

class OrientedSolver(TestingSolver):
    def __init__(self, **params):
        super().__init__(**params)

        self.mode = None

        if self.params['dataset']['dataset_type'] == 'hexagon' or self.params['dataset']['dataset_type'] == 'grid':
            self.mode = 'grid'
            if self.params['dataset']['order_distr'] == 'centered':
                n = self.params['dataset']['n']
                self.center_coords = (n//2,n//2)
            else:
                if self.params['dataset']['order_distr'] == 'airport':
                    n = self.params['dataset']['n']
                    self.center_coords = (0,0)
                else:
                    raise Exception("Not Implemented")

        if self.params['dataset']['dataset_type'] == 'chicago':
            self.mode = 'chicago'
            # loading a dict with actions per census id, with values as dict < target census, probability >
            fm = FileManager("")
            self.to_hub_actions = pkl.load(open(os.path.join(fm.get_all_experiments_data_path(), "to_hub_actions_fixed.pkl"), "rb"))

            self.census_to_node = {}
            for n in self.testing_env.world.nodes(data=True):
                self.census_to_node[n[1]['census']] = n[0]

        if self.mode is None:
            logging.info("Dataset: {}".format(self.params['dataset']))
            raise Exception("Not Implemented")

    def train(self, db_save_callback = None):
        pass

    def load(self):
        pass

    def save(self):
        pass

    def predict(self, observation, info):
        action = np.zeros(self.testing_env.get_action_space_shape())
        wl = len(self.testing_env.world)
        one_cell_action_space = action.shape[0] // wl

        for i in range(wl):
            one_cell_action = np.zeros(one_cell_action_space)
            if self.mode == 'chicago':
                census = self.testing_env.world.nodes[i]['census']
                nonzero = 0
                if census not in self.to_hub_actions:
                    logging.error("Census {} not in hub_to_actions".format(census))
                else:
                    j = 0
                    for nn in self.testing_env.world.neighbors(i):
                        nn_census = self.testing_env.world.nodes[nn]['census']
                        if nn_census in self.to_hub_actions[census]:
                            one_cell_action[j] = self.to_hub_actions[census][nn_census]
                            nonzero += 1
                        j += 1
                    if census in self.to_hub_actions[census]:
                        one_cell_action[-1] = self.to_hub_actions[census][census]
                        nonzero += 1
                    assert nonzero == len(self.to_hub_actions[census])
                if nonzero == 0:
                    one_cell_action[-1] = 1
                action[i*one_cell_action_space:(i+1)*one_cell_action_space] = one_cell_action
            else:
                nn_ranked = []
                j = 0
                self_coords = self.testing_env.world.nodes[i]['coords']
                for nn in self.testing_env.world.neighbors(i):
                    coords = self.testing_env.world.nodes[nn]['coords']
                    rank = abs(self.center_coords[0] - coords[0]) + abs(self.center_coords[1] - coords[1])
                    nn_ranked.append((rank, j))
                    j += 1
                nn_ranked.append((abs(self.center_coords[0] - self_coords[0]) + abs(self.center_coords[1] - self_coords[1]), -1))
                nn_ranked.sort()
                one_cell_action[nn_ranked[0][1]] = 1
                action[i*one_cell_action_space:(i+1)*one_cell_action_space] = one_cell_action

        return action
