import unittest
import os,sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','Simulator','simulator'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','Solvers'))

from Generator import *
from RandomStaticSolver import *
from NoSolver import *
from CitySimple import *
from CityReal import *

class TestEnvironment(unittest.TestCase):

    def setUp(self):
        pass

    def testOneStep(self):
        world_graph = nx.Graph()
        world_graph.add_edges_from([(0,1), (1,2), (2,3), (0,2)])
        idle_driver_locations = np.array([[1, 2, 3, 4], [0, 0, 0, 0]])
        onoff_driver_locations = np.zeros((2, 4, 2))
        n_intervals = 2
        wc = 0.5
        maxincome = None
        real_orders = [(1,2,0,1,20), (2,3,1,1,59)]
        dispatch_actions = [(2,0,2)] # from 2nd to 0th cell - 2 drivers

        simulator = CityReal(world_graph, idle_driver_locations, n_intervals=n_intervals,
                            real_orders=real_orders, onoff_driver_locations=onoff_driver_locations,
                            wc=wc)
        observation = simulator.reset_clean()
        next_state, reward, info = simulator.step(dispatch_actions)

        self.assertEqual(simulator.get_total_revenue(), 20 - (1+2+3+4 - 1)*wc)
        self.assertEqual(simulator.get_min_revenue(), -wc)
        self.assertTrue(np.array_equal(next_state, np.array([[3, 1, 2, 4], [0, 0, 1, 0]])))

    # following deprecated because testing environment was moved from CityReal to gym_taxi

    #
    # def testSimpleEnvironmentRandomPolicy(self):
    #     '''
    #     Create simple random grid networks and test that environment and simple environment give the same result,
    #     for no policy and for random policy.
    #     '''
    #     seeds = [1, 2, 3]
    #     generator_params = {
    #         "dataset_type": "grid",
    #         "n": 4,
    #         "time_periods": 5,
    #         "days": 100,
    #         "orders_density": 0.01,
    #         "number_of_cars": 100,
    #         "order_distr": "uniform"
    #     }
    #
    #     for seed in seeds:
    #         np.random.seed(seed)
    #         gen = Generator("testEnvironment", generator_params)
    #         graph_info = gen.generate()
    #         world_graph, idle_driver_locations, real_orders, onoff_driver_locations, random_average, dist = gen.load_complete_set()
    #
    #         solver_params = {
    #             "dataset": graph_info,
    #             "wc": 0.5,
    #             "count_neighbors": 1,
    #             "tag": "testEnvironment",
    #             "testing_epochs": 2
    #         }
    #         real_env = RandomStaticSolver(**solver_params)
    #         real_env.seed(seed)
    #         real_env.run()
    #
    #         wc = solver_params['wc']
    #         A = real_env.getA()
    #         n_intervals = generator_params['time_periods']
    #         car_distribution = CitySimple.build_car_distribution(idle_driver_locations)
    #         # random_average[0]: random average has averages per each time stamp
    #         small_env = CitySimple(world_graph, wc, random_average[0], car_distribution, dist, A, generator_params["time_periods"])
    #         small_env.run(iterations = generator_params['days'])
    #
    #         self.assertLess(np.abs(real_env.env.get_total_revenue() - small_env.get_total_revenue())/small_env.get_total_revenue(), 0.01)
    #         self.assertEqual(real_env.env.get_min_revenue(), small_env.get_min_revenue())
    #
    # def testRealEnvNoPolicy(self):
    #
    #     generator_params = {
    #         "dataset_type": "grid",
    #         "n": 4,
    #         "time_periods": 5,
    #         "days": 10,
    #         "orders_density": 0.1,
    #         "number_of_cars": 10,
    #         "order_distr": "uniform"
    #     }
    #
    #     gen = Generator("testEnvironment", generator_params)
    #     graph_info = gen.generate()
    #     world_graph, idle_driver_locations, real_orders, onoff_driver_locations, random_average, dist = gen.load_complete_set()
    #
    #     solver_params = {
    #         "dataset": graph_info,
    #         "wc": 0.5,
    #         "tag": "testEnvironment",
    #         "count_neighbors": 1,
    #         "testing_epochs": 2
    #     }
    #     real_env = NoSolver(**solver_params)
    #     real_env.seed(123)
    #     real_env.run()
    #
    #     self.assertNotAlmostEqual(real_env.env.get_total_revenue(), 0)
