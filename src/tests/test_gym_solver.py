import numpy as np

from src.Solvers.GymSolver import GymSolver
from src.Generator import Generator

class TestGymSolver:

    def test_predict(self):
        '''
        Test predict function, that should return multidriver action for multidriver observation,
        given the trained single driver model under stable-baselines framework
        '''

        generator_params = {
            "dataset_type": "grid",
            "n": 3,
            "time_periods": 2, # should work for any network
            "days": 2,
            "orders_density": 10,
            "number_of_cars": 10,
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
            "count_neighbors": 1,
            "weight_poorest": 0,
            "normalize_rewards": 1,
            "minimum_reward": 0,
            "include_income_to_observation": 0,
            "num_cpu": 4,
            "training_iterations": 1000, # check testing while training
            "testing_epochs": 2
        }

        solver = GymSolver(**solver_params)
        solver.train()
        solver.save()

        solver = GymSolver(**solver_params)
        solver.load()
        solver.test()

        solver_params["include_income_to_observation"] = 1
        solver_params["continuous_observation"] = 1
        solver = GymSolver(**solver_params)
        solver.run()
