import numpy as np
import pytest

from framework.solvers.GymSolver import GymSolver
from framework.Generator import Generator

class TestGymSolver:

    def get_solver_params(self, graph_info):
        return {
            "dataset": graph_info,
            "wc": 0.1,
            "tag": "testGymSolver",
            "count_neighbors": 0,
            "weight_poorest": 0,
            "normalize_rewards": 1,
            "gamma": 0.99,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "lam": 0.95,
            "noptepochs": 4,
            "cliprange": 0.2,
            "minimum_reward": 0,
            "include_income_to_observation": 0,
            "num_cpu": 4,
            "callback": 1,
            "draw": 0,
            "idle_reward": 0,
            "hold_observation": 1,
            "save_freq": 1,
            "eval_freq": 1,
            "draw_freq": 0,
            "testing_epochs": 1,
            "seed": 0,
            "robust": 0,
            "robust_nu": 0.1,
            "robust_epsilon": 1,
            "robust_cmin": 0,
            "robust_cmax": 500,
            "robust_gamma": 0.3,
            "debug": 1,
            "training_iterations": 1000, # check testing while training
            "penalty_for_invalid_action": 0,
            "discrete": 0,
            "action_mask": 1,
            "batch_env": 0
        }

    def get_generator_params(self):
        return {
            "dataset_type": "grid",
            "n": 4,
            "time_periods": 10, # should work for any network
            "days": 2,
            "orders_density": 10,
            "number_of_cars": 16,
            "order_distr": "star",
            "order_sampling_multiplier": 1,
            "seed": 0
        }

    @pytest.mark.skip
    def test_basic_training(self):
        '''
        Test if training on trivial parameter set does not fail
        '''
        generator_params = self.get_generator_params()
        gen = Generator("testGymSolver", generator_params)
        graph_info = gen.generate()
        world_graph, idle_driver_locations, real_orders, \
            onoff_driver_locations, random_average, dist = gen.load_complete_set()

        # use OrigSolver as wrapper for params
        solver_params = self.get_solver_params(graph_info)

        solver = GymSolver(**solver_params)
        solver.train()

        solver_params["include_income_to_observation"] = 1
        solver = GymSolver(**solver_params)

        def fake_save_callback(result):
            pass
        solver.run(fake_save_callback)

    # @pytest.mark.skip
    def test_basic_training_batch(self):
        '''
        Test if training on trivial parameter set does not fail 
        if PPO uses TaxiEnvBatch
        '''
        generator_params = self.get_generator_params()
        gen = Generator("testGymSolver", generator_params)
        graph_info = gen.generate()
        world_graph, idle_driver_locations, real_orders, \
            onoff_driver_locations, random_average, dist = gen.load_complete_set()

        # use OrigSolver as wrapper for params
        solver_params = self.get_solver_params(graph_info)
        solver_params['batch_env'] = 1

        solver = GymSolver(**solver_params)
        solver.train()

        solver_params["include_income_to_observation"] = 1
        solver = GymSolver(**solver_params)

        def fake_save_callback(result):
            pass
        solver.run(fake_save_callback)

    ### discrete do not work so far
    @pytest.mark.skip
    def test_discrete(self):
        generator_params = self.get_generator_params()
        gen = Generator("testGymSolver", generator_params)
        graph_info = gen.generate()
        world_graph, idle_driver_locations, real_orders, \
            onoff_driver_locations, random_average, dist = gen.load_complete_set()

        # use OrigSolver as wrapper for params
        solver_params = self.get_solver_params(graph_info)
        solver_params['discrete'] = 1

        solver = GymSolver(**solver_params)
        solver.train()

        solver_params["include_income_to_observation"] = 1
        solver = GymSolver(**solver_params)

        def fake_save_callback(result):
            pass
        solver.run(fake_save_callback)