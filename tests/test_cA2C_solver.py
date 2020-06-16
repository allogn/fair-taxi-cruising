import numpy as np

from framework.solvers.cA2CSolver import cA2CSolver
from framework.solvers.OrigA2CSolver import OrigA2CSolver
from framework.Generator import Generator
from framework.solvers.cA2C.ocA2C import *

class TestcA2CSolver:

    def get_ca2c_params(self, graph_info):
        return {
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
            "draw": 0,
            "seed": 0,
            "debug": 1
        }

    def test_include_observation(self):
        generator_params = {
            "dataset_type": "hexagon",
            "n": 10,
            "time_periods": 2,
            "days": 2,
            "orders_density": 2,
            "number_of_cars": 20,
            "order_distr": "star",
            "order_sampling_multiplier": 1,
            "seed": 0
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
            "order_sampling_multiplier": 1,
            "seed": 0
        }
        ca2c_params = self.get_ca2c_params(graph_info)
        solv = cA2CSolver(**ca2c_params)

        # driver + order dist + income + time + idle_driver
        assert solv.env.get_observation_space_shape() == (4*len(world_graph) + generator_params["time_periods"],)
        observation = solv.env.reset()
        init_info = solv.env.get_reset_info()
        assert observation.shape == solv.env.get_observation_space_shape()
        curr_state, info, income_mat = solv.observation_to_old_fashioned_info(observation, init_info)
        assert (income_mat == np.zeros((len(world_graph),))).all()

        def fake_save_callback(result):
            pass
        solv.run(fake_save_callback)

    def test_view_solver(self):
        # a linear graph with orders and drivers beinng dispatched in over all network
        # only half of the graph is within a view

        generator_params = {
            "dataset_type": "linear",
            "n": 10,
            "time_periods": 2,
            "days": 1,
            "orders_density": 2,
            "number_of_cars": 200,
            "order_distr": "star",
            "order_sampling_multiplier": 1,
            "view_div": 0.1, # view is 10% of nodes (1 node), ordered by node id. 1 node makes max_deg different inside the view
            "seed": 0
        }

        gen = Generator("testTaxiEnvBatch", generator_params)
        graph_info = gen.generate()
        world_graph, idle_driver_locations, real_orders, _, _, _ = gen.load_complete_set()

        # assert drivers ar distributed somehow uniformly
        assert idle_driver_locations.shape == (2,10)
        assert (idle_driver_locations[0] > 0).all()
        # orders are located only on the edges
        orders_sum = np.zeros((10,))
        for r in real_orders:
            orders_sum[r[0]] += 1
        assert orders_sum[0] > 0 and orders_sum[-1] > 0
        assert np.sum(orders_sum[1:-2]) == 0

        ca2c_params = self.get_ca2c_params(graph_info)
        solv = cA2CSolver(**ca2c_params)

        def fake_save_callback(result):
            pass
        solv.run(fake_save_callback)
