import numpy as np

from framework.solvers.cA2CSolver import cA2CSolver
from framework.solvers.OrigA2CSolver import OrigA2CSolver
from framework.Generator import Generator
from framework.solvers.cA2C.ocA2C import *

class TestcA2CSolver:

    def test_include_observation(self):
        generator_params = {
            "dataset_type": "hexagon",
            "n": 10,
            "time_periods": 2,
            "days": 2,
            "orders_density": 2,
            "number_of_cars": 20,
            "order_distr": "star",
            "order_sampling_multiplier": 1
        }

        gen = Generator("testTaxiEnvBatch", generator_params)
        graph_info = gen.generate()
        world_graph, idle_driver_locations, real_orders, \
            onoff_driver_locations, random_average, dist = gen.load_complete_set()

        # use OrigSolver as wrapper for params
        orig_solver_params = {
            "dataset": graph_info,
            "alpha": 0.1,
            "wc": 0,
            "iterations": 1, # 1 epoch
            "tag": "testTaxiEnvBatch",
            "gamma": 0.9,
            "order_sampling_multiplier": 1
        }
        ca2c_params = {
            "dataset": graph_info,
            "wc": 0,
            "iterations": 1, # 1 epoch
            "tag": "testTaxiEnvBatch",
            "epsilon": 0.5,
            "gamma": 0.9,
            "learning_rate": 1e-3,
            "count_neighbors": 1,
            "weight_poorest": 0,
            "normalize_rewards": 1,
            "minimum_reward": 0,
            "batch_size": 20,
            "include_income_to_observation": 1,
            "testing_epochs": 2,
            "draw": 0
        }
        solv = cA2CSolver(**ca2c_params)

        # driver+order dist + income + onehot node id + time
        assert solv.env.get_observation_space_shape() == ((2+3)*len(world_graph) + generator_params["time_periods"],)
        observation = solv.env.reset()
        init_info = solv.env.get_reset_info()
        assert observation.shape == solv.env.get_observation_space_shape()
        curr_state, info, income_mat = solv.observation_to_old_fashioned_info(observation, init_info)
        assert (income_mat == np.zeros((len(world_graph),3))).all()

        solv.run()
