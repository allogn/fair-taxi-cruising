import pytest
import numpy as np
from framework.Generator import Generator
from framework.solvers.OrigNoSolver import OrigNoSolver

class TestOrigSolvers:

    def test_no_solver(self):
        generator_params = {
            "dataset_type": "hexagon",
            "n": 10,
            "time_periods": 144,
            "days": 1,
            "orders_density": 0.1,
            "number_of_cars": 2,
            "order_distr": "uniform"
        }

        np.random.seed(777)
        gen = Generator("testOrigSolvers", generator_params)
        graph_info = gen.generate()
        world_graph, idle_driver_locations, real_orders, onoff_driver_locations, random_average, dist = gen.load_complete_set()

        solver_params = {
            "dataset": graph_info,
            "alpha": 0.1,
            "wc": 0,
            "tag": "testOrigSolvers",
            "gamma": 0.9,
            "train_test_split": 0.5
        }

        solver = OrigNoSolver(**solver_params)
        
        def fake_save_callback(result):
            pass
        solver.run(fake_save_callback)
