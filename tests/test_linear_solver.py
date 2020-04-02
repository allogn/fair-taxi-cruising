import unittest
import os,sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../Solvers'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'Expert'))

import networkx as nx
import numpy as np

from Experiment import *
from LinearSolver import *
from Generator import *
# 
# class TestLinearSolver(unittest.TestCase):
#
#     def testSingleIteration(self):
#         generator_params = {
#             "dataset_type": "grid",
#             "n": 4,
#             "time_periods": 5,
#             "days": 10,
#             "orders_density": 0.5,
#             "number_of_cars": 10,
#             "order_distr": "uniform"
#         }
#
#         np.random.seed(777)
#         gen = Generator("testLinearSolver", generator_params)
#         graph_info = gen.generate()
#         world_graph, idle_driver_locations, real_orders, onoff_driver_locations, random_average, dist = gen.load_complete_set()
#
#         solver_params = {
#             "dataset": graph_info,
#             "tag": "testLinearSolver",
#             "alpha": 0.1,
#             "wc": 0.5,
#             "mode": "Train",
#             "gamma": 0.9,
#             "iterations": 10,
#             "testing_epochs": 2
#         }
#
#         solver = LinearSolver(**solver_params)
#         solver.train()
#         solver.test()
