import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

#from gurobipy import *   -------- not in virtualenv for now
from scipy.spatial.distance import euclidean
from collections import Counter
from numpy.linalg import norm
import time

from TestingSolver import *
from RandomStaticSolver import *

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

np.seterr(divide='ignore')
np.seterr(divide='ignore', invalid='ignore')

class RwSolver(TestingSolver):
    def __init__(self, **params):
        super().__init__(**params)
        self.hours = self.time_periods // self.params["dataset"]["time_periods_per_hour"]

        if params['mode'] == "Test":
            pp = pkl.load(open(os.path.join(self.dpath, "trained_rw_{}.pkl".format(self.params['objective'])), "rb"))
            self.expected_demand, self.Pr, self.A, self.long_term_p = pp

    def get_dispatch_action(self, env, state, context):
        return RandomStaticSolver.sample_actions_based_on_A(self.A, env.world, context)

    def calculate_pr(self, days):
        # learn probabilities Pr from used data
        if self.params['dataset']['time_periods'] % self.params["dataset"]["time_periods_per_hour"] != 0:
            raise Exception("Time periods in a day (time_periods) should be multiple of the time periods per hour")
        self.expected_demand = []
        self.Pr = []
        real_orders = np.array(self.real_orders)
        if len(real_orders) == 0:
            poisson_average = np.zeros((self.N,self.N))
            for hour in range(self.hours):
                self.expected_demand.append(np.sum(poisson_average,axis=1)) # axis=1 => summation of TO values
                self.Pr.append(poisson_average)
            return self.Pr, self.expected_demand

        real_orders = real_orders[np.where(real_orders[:,2] // self.time_periods <= days)]

        for hour in range(self.hours):
            poisson_average = np.zeros((self.N,self.N)) # average per period
            for i in range(self.N):
                for j in range(self.N):
                    cond = np.where(((real_orders[:,2] // self.params["dataset"]["time_periods_per_hour"]) % self.hours == hour) & (real_orders[:,0] == i) & (real_orders[:,1] == j))
                    poisson_average[i,j] = (len(cond[0])/days) / self.params["dataset"]["time_periods_per_hour"]

            self.expected_demand.append(np.sum(poisson_average,axis=1)) # axis=1 => summation of TO values
            self.Pr.append((poisson_average.T / self.expected_demand[-1]).T)
            np.nan_to_num(self.Pr[-1],copy=False)
        return self.Pr, self.expected_demand

    def train(self):
        all_train_time = time.time()
        t1 = 0
        self.world = nx.read_gpickle(os.path.join(self.dpath, "world.pkl"))
        with open(os.path.join(self.dpath, "dist.pkl"), "rb") as f:
            self.dist = pkl.load(f)

        self.N = len(self.world)
        self.real_orders, _, _ = self.get_train_data()
        _, self.idle_driver_locations, _ = self.get_test_data()
        self.log['loading_time'] = time.time() - t1

        # calculate Pr or load true values
        if self.params['set_true_pr'] == 1:
            random_average = pkl.load(open(os.path.join(self.params['dataset']['dataset_path'], "random_average_original.pkl"), "rb"))
            self.expected_demand = []
            self.Pr = []
            for hour in range(self.hours):
                self.expected_demand.append(np.sum(random_average[hour],axis=1)) # axis=1 => summation of TO values
                self.Pr.append((random_average[hour].T / self.expected_demand[-1]).T)
        else:
            t1 = time.time()
            self.calculate_pr(self.first_test_day)
            self.log["Pr estimation time"] = time.time() - t1
        self.expected_demand = np.array(self.expected_demand)

        A, p = self.train_long_term_policy()
        self.log['train_time'] = time.time() - all_train_time
        with open(os.path.join(self.dpath, "trained_rw_{}.pkl".format(self.params['objective'])), "wb") as f:
            pkl.dump([self.expected_demand, self.Pr, A, p], f)

    def train_long_term_policy(self):
        number_of_cars = self.params['dataset']['number_of_cars']
        p_init = np.zeros((self.time_periods, number_of_cars, self.N))

        return RandomStaticSolver.get_random_A(self.world), p_init

        raise Exception("Not implemented: improving policy should be updated to support multiperiods")
        t1 = time.time()
        self.log['expected_objective'] = []
        self.log['expected_objective2'] = []
        assert(number_of_cars > 0)

        filled_car = 0
        for i in range(self.idle_driver_locations.shape[1]):
            while self.idle_driver_locations[0][i] > 0:
                p_init[0][filled_car][i] += 1
                filled_car += 1
                self.idle_driver_locations[0][i] -= 1

        A_init = RandomStaticSolver.get_random_A(self.world)

        beta = self.params['beta']
        delta = self.params['delta']
        wc = self.params['wc']
        t1 = time.time()
        p, probability_of_busy, A = self.get_transition_probability_and_improve_policy(self.params["iterations"],
                                    beta, self.Pr, self.expected_demand, p_init, A_init,
                                    number_of_cars, delta, wc, self.world, self.dist)

        return A, p

    @staticmethod
    def improve_policy_iteration(objective, beta, A, coef_dict, coef_slag, number_of_cars, N, variables):
        m = Model('taxi')
        m.setParam('OutputFlag', 0)
        policy = m.addVars(variables, lb=0.0, ub=1.0, name="policy", vtype=GRB.CONTINUOUS)
        m.addConstrs((policy.sum(i,'*') == 1 for i in range(N)), "cap")

        if objective == "min":
            z = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z")
            m.setObjective(z, GRB.MAXIMIZE)
            costr = m.addConstrs((policy.prod(coef_dict[q]) + coef_slag[q] - z >= 0 for q in range(number_of_cars)), "cap")
        if objective == "total":
            m.setObjective(quicksum([policy.prod(coef_dict[q]) + coef_slag[q] for q in range(number_of_cars)]), GRB.MAXIMIZE)

        m.optimize()

        status = m.getAttr('Status')
        if status == 3:
            raise Exception("Infeasible LP model")
        if status == 4:
            raise Exception("Unbounded or infeasible LP model")
        if status != 2:
            raise Exception("Unknown status of LP: {}".format(status))

        solution = m.getAttr('X', policy)
        for var in solution:
            A[var[0], var[1]] = solution[var]*beta + A[var[0], var[1]]*(1-beta)
        A = (A.T / np.sum(A,axis=1)).T
        return A

    @staticmethod
    def get_transition_probability_iteration(Pr, A, expected_demand, p, number_of_cars, delta):
        avg_p = np.mean(p, axis=0)

        # inverese of expected cars per cell
        reciprocal = np.divide(1-np.power(1-avg_p, number_of_cars), number_of_cars*avg_p)
        reciprocal[np.where(avg_p == 0)] = 1

        N = A.shape[0]
        assert(expected_demand.shape == (N,))
        probability_of_busy = np.multiply(expected_demand, reciprocal)
        probability_of_busy[np.where(probability_of_busy > 1)] = 1
        # softmax(probability_of_busy)
        B = (Pr.T * probability_of_busy).T + (A.T * (1-probability_of_busy)).T
        assert(np.sum(B,axis=1).all() == 1)
        assert(B.shape == (N,N))

        phi = np.dot(p, B)
        assert(phi.shape == (number_of_cars,N))
        assert((np.abs(np.sum(phi,axis=1) - 1) < 0.00001).all())
        p = (1-delta)*p + delta*phi
        assert((np.abs(np.sum(p,axis=1) - 1)< 0.00001).all())
        return p, probability_of_busy

    @staticmethod
    def get_transition_probability(iterations, Pr, A, expected_demand, p_init, number_of_cars, delta):
        '''
        p_init: matrix MxN, M - cars, N - cells. One-Hot of where cars are initially located.
        '''
        N = p_init.shape[1]
        assert(np.sum(Pr,axis=1).all() == 1)
        p = np.copy(p_init)
        for t in range(iterations):
            p, probability_of_busy = RwSolver.get_transition_probability_iteration(Pr, A, expected_demand, p, number_of_cars, delta)
        return p, probability_of_busy

    @staticmethod
    def get_transition_probability_and_improve_policy(iterations, beta, Pr, expected_demand, p_init, A_init, number_of_cars, delta, wc, world, dist):
        N = p_init.shape[1]
        A = np.copy(A_init)

        # for pt in range(policy_iterations):
        p = np.copy(p_init)
        for t in range(iterations):
            p, probability_of_busy = RwSolver.get_transition_probability_iteration(Pr, A, expected_demand, p, number_of_cars, delta)

            variables, coef_dict, coef_slag = RwSolver.get_lp(p, probability_of_busy, Pr, A, expected_demand, number_of_cars, dist, wc, world)
            A = RwSolver.improve_policy_iteration("total", beta, A, coef_dict, coef_slag, number_of_cars, N, variables)

        # p = np.copy(p_init)
        # for t in range(iterations):
        #     p, probability_of_busy = RwSolver.get_transition_probability_iteration(Pr, A, expected_demand, p, number_of_cars, delta)

        return p, probability_of_busy, A

    @staticmethod
    def get_lp(p, probability_of_busy, Pr, A, expected_demand, number_of_cars, dist, wc, world):
        H = np.sum(np.multiply(Pr, dist), axis=1) * probability_of_busy - wc * (1-probability_of_busy)
        coef_slag = np.dot( np.dot(p, (Pr.T * probability_of_busy).T), H)

        variables = []
        for i in range(len(world)):
            variables.append((i, i))
            for possibility in world.neighbors(i):
                variables.append((i, possibility))

        coef_dict = []
        for q in range(number_of_cars):
            coef_dict.append({})
            for var in variables:
                k = var[0]
                l = var[1]
                coef_dict[-1][var] = float(p[q, k] * (1-probability_of_busy[k]) * H[l])

        return variables, coef_dict, coef_slag

    @staticmethod
    def get_expected_revenue(p, probability_of_busy, Pr, A, expected_demand, number_of_cars, dist, wc, world):
        variables, coef_dict, coef_slag = RwSolver.get_lp(p, probability_of_busy, Pr, A, expected_demand, number_of_cars, dist, wc, world)
        expected_objective = np.sum([np.sum([coef_dict[q][v]*A[v[0],v[1]] for v in variables]) + coef_slag[q] for q in range(number_of_cars)])
        return expected_objective

    @staticmethod
    def get_expected_revenue_per_cell(p, probability_of_busy, Pr, dist, wc):
        expected_cars = np.sum(p, axis=0)
        # print(expected_cars)
        # print((np.sum(np.multiply(Pr, dist), axis=1) * probability_of_busy - wc * (1-probability_of_busy)))
        return expected_cars * (np.sum(np.multiply(Pr, dist), axis=1) * probability_of_busy - wc * (1-probability_of_busy))

    def load(self):
        pass # refactor eventually

    def save(self):
        pass # refactor eventually
