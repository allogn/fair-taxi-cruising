import numpy as np

from src.Solvers.OrientedSolver import OrientedSolver
from src.Generator import Generator

class TestOrientedSolver:

    def test_chicago(self):
        generator_params = {
            "dataset_type": "chicago",
            "days": 5,
            "number_of_cars": 10,
            'order_sampling_multiplier': 1
        }
        gen = Generator("testOrientedSolver", generator_params)
        graph_info = gen.generate()
        world_graph, idle_driver_locations, real_orders, \
            onoff_driver_locations, random_average, dist = gen.load_complete_set()

        solver_params = {
            "dataset": graph_info,
            "wc": 0.1,
            "tag": "testOrientedSolver",
            "count_neighbors": 1,
            "weight_poorest": 0,
            "normalize_rewards": 1,
            "minimum_reward": 0,
            "include_income_to_observation": 0,
            "training_iterations": 10,
            "testing_epochs": 2
        }

        solver = OrientedSolver(**solver_params)
        solver.run()
