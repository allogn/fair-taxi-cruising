import unittest
import os,sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../Solvers'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'Expert'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'Simulator', 'simulator'))

import logging
from numpy.linalg import norm

from Experiment import *
from RandomStaticSolver import *
from RwSolver import *
from CityReal import *
from CitySimple import *

class TestRwSolver(unittest.TestCase):

    def setUp(self):
        pass

    def testTransitionProbabilityIteration(self):
        scale = 0.3
        n = 5
        mc_iterations = 1000
        delta = 0.5
        wc = 0.3
        seed = 10

        world = nx.grid_2d_graph(n, n)
        for v in world.nodes():
            world.nodes[v]['coords'] = (v[0], v[1])
        world = nx.convert_node_labels_to_integers(world)

        N = n*n
        dist = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                dist[i,j] = euclidean(world.nodes[i]['coords'], world.nodes[j]['coords'])
                dist[j,i] = dist[i,j]

        for number_of_cars in [100, 200]:
            np.random.seed(seed)
            random_average = np.random.random(size=(N,N))*scale
            Pr = np.divide(random_average.T, np.sum(random_average,axis=1)).T
            car_distribution = np.random.choice(N, number_of_cars)
            p_init = np.zeros((number_of_cars, N))
            for i in range(len(car_distribution)):
                p_init[i][car_distribution[i]] = 1

            A = RandomStaticSolver.get_random_A(world)

            expected_demand = np.sum(random_average, axis=1)
            p_estimation, p_busy = RwSolver.get_transition_probability_iteration(Pr, A, expected_demand, p_init, number_of_cars, 1)

            target_locations = np.zeros((number_of_cars, N))

            car_distribution_orig = np.copy(car_distribution)

            for i in range(mc_iterations):
                simulator = CitySimple(world, wc, random_average, car_distribution, dist, A, 1)
                simulator.step()
                target_locations += simulator.target_locations

            p_mc = target_locations / mc_iterations
            p_diff = np.ones(p_mc.shape)
            p_diff = (p_diff.T / np.sum(p_diff, axis=1)).T
            error = norm(p_mc - p_estimation)
            error2 = norm(p_mc - p_diff)
            logging.info("Single iteration errors between Diff {} and RW {}".format(error2, error))

    # def testTransitionProbabilityIndependent(self):
    #     scale = 0.3
    #     n = 5
    #     mc_iterations = 1000
    #     delta = 0.5
    #     wc = 0.3
    #     seed = 10
    #
    #     world = nx.grid_2d_graph(n, n)
    #     for v in world.nodes():
    #         world.nodes[v]['coords'] = (v[0], v[1])
    #     world = nx.convert_node_labels_to_integers(world)
    #
    #     N = n*n
    #     dist = np.zeros((N, N))
    #     for i in range(N):
    #         for j in range(i, N):
    #             dist[i,j] = euclidean(world.nodes[i]['coords'], world.nodes[j]['coords'])
    #             dist[j,i] = dist[i,j]
    #
    #     for number_of_cars in [100, 200]:
    #         np.random.seed(seed)
    #         random_average = np.random.random(size=(N,N))*scale
    #         Pr = np.divide(random_average.T, np.sum(random_average,axis=1)).T
    #         car_distribution = np.random.choice(N, number_of_cars)
    #         p_init = np.zeros((number_of_cars, N))
    #         for i in range(len(car_distribution)):
    #             p_init[i][car_distribution[i]] = 1
    #
    #         A = RandomStaticSolver.get_random_A(N)
    #
    #         expected_demand = np.sum(random_average, axis=1)
    #         p_estimation, p_busy = RwSolver.get_transition_probability(mc_iterations, Pr, A, expected_demand, p_init, number_of_cars, 1)
    #
    #         target_locations = np.zeros((number_of_cars, N))
    #
    #         car_distribution_orig = np.copy(car_distribution)
    #
    #         simulator = CitySimple(world, wc, random_average, car_distribution, dist, A)
    #         simulator.run()
    #         target_locations = simulator.target_locations
    #
    #         p_mc = target_locations / mc_iterations
    #         p_diff = np.ones(p_mc.shape)
    #         p_diff = (p_diff.T / np.sum(p_diff, axis=1)).T
    #         error = norm(p_mc - p_estimation)
    #         error2 = norm(p_mc - p_diff)
    #         print("Errors Diff {} vs RW {}".format(error2, error))
    #
    # def testDiffPosToDiffDistr(self):
    #     '''
    #     Test that different initial car positions lead to different probabilistic distributions
    #     '''
    #
    #     number_of_cars = 100
    #     max_demand = 3
    #     n = 5
    #     seed = 100
    #     iterations = 10000 # with 100 000 error very small
    #     delta = 0.5
    #     attempts = 3
    #
    #     world = nx.grid_2d_graph(n, n)
    #     world = nx.convert_node_labels_to_integers(world)
    #     np.random.seed(seed)
    #     N = n*n
    #
    #     estimations = []
    #     for i in range(attempts):
    #         print(i)
    #         random_average = np.random.randint(0,max_demand,size=(N,N))
    #         Pr = np.divide(random_average.T, np.sum(random_average,axis=1)).T
    #         car_distribution = np.random.choice(N, number_of_cars)
    #         p_init = np.zeros((number_of_cars, N))
    #         for i in range(len(car_distribution)):
    #             p_init[i][car_distribution[i]] = 1
    #
    #         A = np.zeros((N,N))
    #         for i in range(N):
    #             A[i][i] = np.random.random()
    #             for j in world.neighbors(i):
    #                 A[i][j] = np.random.random()
    #         A = (A.T / np.sum(A,axis=1)).T
    #
    #         p_estimation, p_busy = RwSolver.get_transition_probability(iterations, Pr, A, random_average, p_init, number_of_cars, delta)
    #         estimations.append(p_estimation)
    #
    #     estimations = np.array(estimations)
    #     avg_est = np.mean(estimations,axis=0)
    #     p_diff = np.ones(avg_est.shape)
    #     p_diff = (p_diff.T / np.sum(p_diff, axis=1)).T
    #
    #     diff = [norm(estimations[i] - avg_est) for i in range(len(estimations))]
    #     print(norm(p_diff - avg_est), diff, np.mean(diff))
    #
    # def testAvgRevenuePerCell(self):
    #     p = np.array([[0.1, 0.2, 0.7], [0.3, 0.2, 0.5]])
    #     probability_of_busy = np.array([0.1, 0.2, 0.3])
    #     Pr = np.ones((3,3))/3
    #     dist = np.ones((3,3))
    #     wc = 2
    #     avg_rev = RwSolver.get_expected_revenue_per_cell(p, probability_of_busy, Pr, dist, wc)
    #     print(avg_rev)
