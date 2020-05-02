import os, sys
import networkx as nx
from collections import Counter
import math
from hilbertcurve.hilbertcurve import HilbertCurve

from framework.solvers.TestingSolver import *
from framework.solvers.cA2CSolver import *
from framework.solvers.DiffSolver import *

class SplitSolver(TestingSolver):
    def __init__(self, **params):
        super().__init__(**params)
        for n in self.world.nodes(data=True):
            assert 'coords' in n[1], "Graph must contain coords"
        superworld_size = (1. - self.params['shrinking_fraction']) * len(self.world)
        assert superworld_size > 0
        self.superworld = self.build_superworld(self.world, superworld_size)
        
    @staticmethod # bug: not necesarilly connected graph
    def build_superworld(world, new_size):
        p = int(math.log2(len(world))) + 1 # 2^p is number of points along each dimention of the space
        hc = HilbertCurve(p, 2)
        node_seq_coords = [(hc.distance_from_coordinates(n[1]['coords']), n[0]) for n in world.nodes(data=True)]
        node_seq_coords.sort()
        chunks = np.array_split(node_seq_coords, new_size)
        center_nodes = [chunk[len(chunk)//2][1] for chunk in chunks]
        node_mapping = nx.voronoi_cells(world, center_nodes)

        new_nodes = []
        i = 0
        for center_node in center_nodes:
            new_nodes.append(
                (i, {"nodes": node_mapping[center_node]})
            )
            for n in node_mapping[center_node]:
                world.nodes[n]['supernode'] = i
            i += 1
        superworld = nx.Graph() # we always use graph to simplify training, for both world and superworld
        superworld.add_nodes_from(new_nodes)
        # we assume all areas are approximately the same size, so we add edges in superworld of the same length
        edges = set()
        for e in world.edges():
            n1 = world.nodes[e[0]]['supernode'] 
            n2 = world.nodes[e[1]]['supernode']
            if n1 != n2 and (n1, n2) not in edges:
                edges.add((n1, n2))
        superworld.add_edges_from(list(edges))

        return superworld

    def find_car_fraction_to_dest(self):
        '''
        This finds what is the fraction of idle cars that needs to be send due to order density
        together with the destinations
        '''
        bigraph = self.build_bipartite_graph()
        idle_dispatch_list = self.solve_bimatching(bigraph)

    def build_bipartite_graph(self):
        '''
    	Build a bigraph between orders and destinations
        with supply/demand equal to the excess of orders in an area.
        On one side - currently existing distribution of drivers
        On another side - a distribution that corresponds to the orders.
        '''
        expected_drivers = self.get_expected_drivers_per_supernode()
        expected_orders = self.get_expected_orders_per_supernode()

        return None

    def get_expected_drivers_per_supernode(self):
        '''
        Expected drivers are drivers that are currently idle in the area
        + drivers that will arrive there in the next iteration
        '''
        ...

    def get_expected_orders_per_supernode(self):
        '''
        @return 
        '''

    def solve_bimatching(self, bigraph):
        '''
        Find how many cars should go where, based on the bigraph
        @return number of cars with source and destinations
        '''
        return None

    def predict(self, observation, info):
        # create a manager

        self.find_car_fraction_to_dest()
        # dispatch partially, by updating the observation

        # for each idle driver - manager fisrt takes care of the correct distribution


        # manager sends closest idle drivers so to fix the distribution


        # within each cell ask cA2C to solve the rest

        action = np.ones(self.testing_env.get_action_space_shape())
        action /= action.shape[0]
        return action

    def train(self, db_save_callback = None):
        # split a world into different worlds based on order logs

        # create cA2C per each world

        # train each cA2C using an estimate of average drivers, and adding drivers in random

        # make sure the model is saved

        pass

    def load(self):
        pass

    def save(self):
        pass
