import numpy as np

from src.Solvers.cA2CSolver import cA2CSolver
from src.Solvers.OrigA2CSolver import OrigA2CSolver
from src.Generator import Generator
from src.Simulator.algorithm.ocA2C import *

class TestcA2CSolver:

    def test_init(self):
        generator_params = {
            "dataset_type": "hexagon",
            "n": 10,
            "time_periods": 144,
            "days": 2,
            "orders_density": 100,
            "number_of_cars": 110,
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
            "gamma": 0.9
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
            "batch_size": 2000,
            "include_income_to_observation": 0,
            "testing_epochs": 2
        }
        origSolv = OrigA2CSolver(**orig_solver_params)
        solv = cA2CSolver(**ca2c_params)

        init_observation = origSolv.env.reset_clean() # observation here is just driver and customer distributions
        temp = np.array(origSolv.env.target_grids) + origSolv.env.M * origSolv.env.N
        target_id_states = origSolv.env.target_grids + temp.tolist()
        stateprocessor = stateProcessor(target_id_states, origSolv.env.target_grids,origSolv.env.n_valid_grids)
        curr_s = stateprocessor.utility_conver_states(init_observation)
        normalized_init_observation = stateprocessor.utility_normalize_states(curr_s)
        init_observation2 = solv.env.reset() # observation here is a full set of drivers, customers, and context

        assert (normalized_init_observation.flatten()[:len(world_graph)] == init_observation2[:len(world_graph)]).all()
        mask = np.ones(len(world_graph))
        mask[0] = 0
        mask[9] = 0
        mask[99] = 0
        mask[90] = 0
        A = normalized_init_observation.flatten()[len(world_graph):2*len(world_graph)] * mask
        B = init_observation2[len(world_graph):2*len(world_graph)] * mask
        assert (A == B).all() # might be a bit different on the corners as there is order sampling involved

        solv.train()
        origSolv.train()

    def test_include_observation(self):
        generator_params = {
            "dataset_type": "hexagon",
            "n": 10,
            "time_periods": 2,
            "days": 2,
            "orders_density": 2,
            "number_of_cars": 200,
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
            "testing_epochs": 2
        }
        solv = cA2CSolver(**ca2c_params)

        # driver+order dist + income + onehot node id + time
        assert solv.env.observation_space_shape == ((2+3)*len(world_graph) + generator_params["time_periods"],)
        observation = solv.env.reset()
        init_info = solv.env.get_reset_info()
        assert observation.shape == solv.env.observation_space_shape
        curr_state, info, income_mat = solv.observation_to_old_fashioned_info(observation, init_info)
        assert (income_mat == np.zeros((len(world_graph),3))).all()

        solv.run()
