import unittest
import os,sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../Solvers'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'Expert'))

import networkx as nx
import numpy as np

from Experiment import *
from ValIterSolver import *

class TestValIterSolver(unittest.TestCase):

    def setUp(self):
        pass
