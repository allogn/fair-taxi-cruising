import unittest
import numpy as np
from framework.Generator import *

class TestGenerator(unittest.TestCase):

    def setUp(self):
        self.random = np.random.RandomState(12345)

    def test_star_distribution(self):
        n = 5

        G = nx.grid_2d_graph(n, n)
        for n in G.nodes():
            G.nodes[n]['coords'] = (n[0], n[1])
        G = nx.convert_node_labels_to_integers(G)

        star_average = Generator.get_random_average_orders("star", 0.5, G, self.random)
        # 0  1  2  3  4
        # 5  6  7  8  9
        # 10 11 12 13 14
        # 15 16 17 18 19
        # 20 21 22 23 24
        result = np.zeros((25, 25))
        result[0, 12] = 0.5
        result[4, 12] = 0.5
        result[20, 12] = 0.5
        result[24, 12] = 0.5
        assert np.array_equal(star_average, result)

    def testLinearDistribution(self):
        n = 5
        G = nx.Graph()
        G.add_edges_from([(0,1),(1,2),(2,3),(3,4)])
        for n in G.nodes():
            G.nodes[n]['coords'] = (n, 0)
        star_average = Generator.get_random_average_orders("star", 0.45, G, self.random)
        result = np.zeros((5, 5))
        result[0, 2] = 0.45
        result[4, 2] = 0.45
        assert np.array_equal(star_average, result)

    def testChicagoSampling(self):
        # load original graph
        params_original = {
            "dataset_type": "chicago",
            "days": 10,
            "sparsity": 1,
            "driver_sampling_multiplier": 1,
            "seed": 0
        }
        gen = Generator("testChicago", params_original)
        gen.generate()
        graph_orig, idle_driver_locations, real_orders, onoff_driver_locations, _, _ = gen.load_complete_set()

        params = {
              "dataset_type": "chicago",
              "days": 10,
              "sparsity": 3,
              "driver_sampling_multiplier": 1,
              "seed": 0
            }

        gen = Generator("testChicago", params)
        gen.generate()
        graph, idle_driver_locations, real_orders, onoff_driver_locations, _, _ = gen.load_complete_set()
        real_orders = np.array(real_orders)
        exp_size = (len(graph_orig)-1) // params['sparsity'] + 1
        self.assertEqual(idle_driver_locations.shape, (24*60//15, exp_size))
        self.assertEqual(len(graph), exp_size)
        self.assertGreater(np.sum(np.array(real_orders)[:1,:]), 0)
        self.assertGreater(np.sum(idle_driver_locations), 0)

    def test_hexagon(self):
        params = {
            "dataset_type": "hexagon",
            "days": 10,
            "n": 3,
            "time_periods": 4,
            "order_distr": "uniform",
            "orders_density": 1,
            "number_of_cars": 10,
            "seed": 0
        }
        gen = Generator("testGenerator", params)
        gen.generate()
        g, _, _, _, _, _ = gen.load_complete_set()
        assert g.has_edge(0,1)
        assert g.has_edge(0,4)
        assert g.has_edge(1,3)
        assert g.has_edge(1,5)
        assert g.number_of_edges() == 20

    def test_matthew(self):
        m = Generator.generate_matthew_matrix(4, 81, self.random, 1)
        assert m.shape == (81,81)
        allowed_rows = [0, 3, 3*9, 3*9+3, 80, 77, 81-3*9-1, 81-3*9-4]
        for i in range(81):
            if i in allowed_rows:   
                assert np.sum(m[i,:]) > 0, i
            else:
                assert np.sum(m[i,:]) == 0, i

    def test_matthew_target(self):
        m = Generator.get_random_matthew_target(4, 81, self.random, 1, False)
        assert (m[:4] > 0).all()
        assert (m[4:9] == 0).all()
        assert (m[9:13] > 0).all()
        assert (m[13:18] == 0).all()
        assert (m[18:22] > 0).all()
        assert (m[22:27] == 0).all()
        assert (m[27:31] > 0).all()
        assert (m[31:] == 0).all()
        assert m.shape == (81,)
