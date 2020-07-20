import numpy as np
import networkx as nx

from framework.solvers.SplitSolver import SplitSolver
from framework.Generator import Generator

class TestSplitSolver:

    def test_build_worlds(self):
        n = 6
        G = nx.grid_2d_graph(n, n, create_using=nx.DiGraph())
        for n in G.nodes():
            G.nodes[n]['coords'] = (n[0], n[1])
        G = nx.convert_node_labels_to_integers(G)
        superworld = SplitSolver.build_superworld(G, 4)
        assert len(superworld) == 4
        for n in superworld.nodes(data=True):
            assert abs(len(n[1]['nodes']) - 9) <= 3 # might not be equal
            for n2 in n[1]['nodes']:
                assert G.has_node(n2)
            assert nx.is_weakly_connected(G.subgraph(n[1]['nodes']))
        assert superworld.number_of_edges() >= 4
        # might be greater than 8, because splitting might not be exactly square-based (due to hilbert approximation)
        for n in G.nodes(data=True):
            assert 'supernode' in n[1]

    def test_flow_per_edge(self):
        g = nx.complete_graph(5)

        # test excess drivers
        idle_drivers = [0, 0, 10, 0, 23]
        arriving_drivers = [0, 0, 0, 0, 0] 
        expected_orders = [2, 2, 0, 0, 0]
        flow_per_edge = SplitSolver.get_flow_per_edge(idle_drivers, arriving_drivers, expected_orders, g)
        s = [0, 0, 10, 0, 23]
        for source, val in flow_per_edge.items():
            for dest, flow in val.items():
                assert flow >= 0
                s[source] -= flow
                s[dest] += flow
        assert s[0] == 17 or s[0] == 16
        assert s[1] == 16 or s[1] == 17
        assert s[0] + s[1] == np.sum(idle_drivers)
        assert np.sum(s[2:]) == 0

        # test arriving drivers
        idle_drivers = [5, 0, 0, 0, 5]
        arriving_drivers = [0, 10, 0, 10, 0] 
        expected_orders = [0, 0, 0, 0, 0]
        flow_per_edge = SplitSolver.get_flow_per_edge(idle_drivers, arriving_drivers, expected_orders, g)
        s = np.array(idle_drivers)
        for source, val in flow_per_edge.items():
            for dest, flow in val.items():
                assert flow >= 0
                s[source] -= flow
                s[dest] += flow
        assert np.sum(s) == np.sum(idle_drivers), (s, flow_per_edge)
        assert s[1] == 0 and s[3] == 0 and s[0] > 1 and s[2] >= 1 and s[4] >= 1, (s, flow_per_edge)

        # test non-complete graph
        # ...

    def test_init_solvers(self):
        # create simple environment
        gen = Generator("test")
        graph_info = gen.generate()
        params = {
            "dataset": graph_info,
            "wc": 0,
            "iterations": 2,
            "tag": "test",
            "count_neighbors": 1,
            "weight_poorest": 0,
            "normalize_rewards": 1,
            "minimum_reward": 0,
            "poorest_first": 1,
            "include_income_to_observation": 0,
            "testing_epochs": 2,
            "draw": 0,
            "shrinking_fraction": 0.9,
            "subsolver": "Diff",
            "debug": 1,
            "seed": 123,
            'penalty_for_invalid_action': 1000,
            "discrete": 0
        }
        solv = SplitSolver(**params)
        solv.init_subsolvers()