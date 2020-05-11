import numpy as np

from framework.solvers.GymSolver import GymSolver
from framework.Generator import Generator

class TestGymSolver:

    def test_basic_training(self):
        '''
        Test if training on trivial parameter set does not fail
        '''

        generator_params = {
            "dataset_type": "grid",
            "n": 4,
            "time_periods": 10, # should work for any network
            "days": 2,
            "orders_density": 10,
            "number_of_cars": 16,
            "order_distr": "star",
            "order_sampling_multiplier": 1
        }


        gen = Generator("testGymSolver", generator_params)
        graph_info = gen.generate()
        world_graph, idle_driver_locations, real_orders, \
            onoff_driver_locations, random_average, dist = gen.load_complete_set()

        # use OrigSolver as wrapper for params
        solver_params = {
            "dataset": graph_info,
            "wc": 0.1,
            "tag": "testGymSolver",
            "count_neighbors": 0,
            "weight_poorest": 0,
            "normalize_rewards": 1,
            "minimum_reward": 0,
            "include_income_to_observation": 0,
            "num_cpu": 4,
            "callback": 1,
            "draw": 0,
            "save_freq": 1,
            "eval_freq": 1,
            "testing_epochs": 1,
            "training_iterations": 1000 # check testing while training
        }

        solver = GymSolver(**solver_params)
        solver.train()

        solver_params["include_income_to_observation"] = 1
        solver = GymSolver(**solver_params)

        def fake_save_callback(result):
            pass
        solver.run(fake_save_callback)
