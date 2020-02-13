import numpy as np

from src.Solvers.Solver import Solver
from src.Generator import Generator
from src.Simulator.simulator.oenvs import CityReal

class OrigSolver(Solver):
    '''
    Class serves as wrapper of original code:
    Now we can use Generator class and Experimental framework to generate special cases to run via original code
    '''

    def __init__(self, **params):
        super().__init__(**params)

        # parameters from our config, not the original one
        self.days = self.params['dataset']["days"]
        assert self.params['wc'] == 0, "Only zero-cost movements supported"
        self.load_dataset()
        self.env = CityReal(self.mapped_matrix_int, self.order_num_dist,
                               self.idle_driver_dist_time, self.idle_driver_location_mat,
                               self.order_time_dist, self.order_price_dist,
                               self.l_max, self.M, self.N, self.n_side, self.order_sample_p, self.order_real, self.onoff_driver_location_mat,
                               global_flag="")

    def load_dataset(self):
        '''
        load complete dataset

        note that orders are merged into a single day, and then sampled out of there
        '''
        dataset_params = self.params['dataset']
        gen = Generator(self.params['tag'], dataset_params)
        assert dataset_params['dataset_type'] == 'hexagon', "Only hexagon dataset supported"
        world, idle_driver_locations, real_orders, \
            onoff_driver_locations, random_average, dist = gen.load_complete_set(dataset_id=self.params['dataset']['dataset_id'])

        self.l_max = 9 # default 9 (1.5h orders) can be served max. should not matter if orders are "real" and not generated

        self.n_side = 6 # number of neighbors to travel
        self.M = dataset_params['n']
        self.N = dataset_params['n']
        self.mapped_matrix_int = np.reshape(np.arange(0,len(world)), (self.N, self.M)) # should be positive for some reason
        self.order_num_dist = None # should be used only for synthetic orders, we always generate orders ourselves
        self.order_time_dist = None
        self.order_price_dist = None
        self.idle_driver_dist_time = None # [time, mean, std] -- we generate total number of drivers not randomly, but load them from generator
        self.idle_driver_location_mat = idle_driver_locations
        self.onoff_driver_location_mat = onoff_driver_locations

        # collect all orders in one day and sample them
        self.order_real = np.array(real_orders)
        for i in np.arange(len(self.order_real)):
            self.order_real[i][2] = self.order_real[i][2] % 144 # merge all together

        self.order_sample_p = 1./self.days

    def load(self):
        pass # refactor eventually

    def save(self):
        pass # refactor eventually
