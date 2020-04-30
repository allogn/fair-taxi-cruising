import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from TestingSolver import *
from RwSolver import *
from cA2CSolver import * # required for state processor
from ValIterSolver import *

#TODO fix rewards (from experimetns)- constant value for some baselines.
class LinearSolver(TestingSolver):
    def __init__(self, **params):
        '''
        alpha parameter is convergence rate, gamma parameter is to pass to valueIter for value function learning
        '''
        super().__init__(**params)
        t1 = time.time()
        self.alpha = self.params['alpha']

        dpath = self.params['dataset']["dataset_path"]
        self.log_dir = dpath
        time_periods = self.params['dataset']["time_periods"]
        self.world = nx.read_gpickle(os.path.join(dpath, "world.pkl"))

        if params['mode'] == "Train":
            self.real_orders, self.idle_driver_locations, self.onoff_driver_locations = self.get_train_data()
        else:
            self.real_orders, self.idle_driver_locations, self.onoff_driver_locations = self.get_test_data()

        self.log['init_time'] = time.time() - t1
        self.state_size = len(self.world)*5 # in linear_state() another definition
        self.build_policy_mapping()
        self.init()

    def build_policy_mapping(self):
        mapping = {}
        i = 0
        for node in self.world.nodes():
            neighbors = list(self.world.neighbors(node)) + [node]
            mapping[node] = list(range(i,i+len(neighbors)))
            i += len(neighbors)

        weights = np.ones((self.state_size,i))
        self.node_to_output_index = mapping
        self.weights = weights

    @staticmethod
    def get_linear_state(world, normalized_state, context):
        '''
        normalized state - [drivers, customers] (normalized)
        context - relu([drivers-custer, cust-drivers])
        '''
        N = len(world)
        state_size = len(world)*5
        state = np.zeros(state_size)
        for n in world.nodes():
            node = world.nodes[n]['o']
            if node.get_driver_numbers() > 0:
                state[n] = np.mean([node.drivers[d].get_income() for d in node.drivers])
        state[N:3*N] = normalized_state
        state[3*N:5*N] = context
        return state

    def init(self):
        self.env = CityReal(self.world, self.idle_driver_locations, real_orders=self.real_orders,
                            onoff_driver_locations=self.onoff_driver_locations, n_intervals=self.time_periods, wc=self.params["wc"])
        self.env.reset_randomseed(self.random_seed)

    def action(self, world_state):
        N = len(self.world)
        output = np.dot(world_state, self.weights)
        required_actions = world_state[3*N:4*N]
        action_tuples = []
        output_multiplicity = np.zeros(output.shape)
        for n in self.world.nodes():
            idle_cars = required_actions[n]
            output_index = self.node_to_output_index[n]
            pvec = output[output_index]

            pvec = softmax(pvec)
            assert(pvec.shape == (len(output_index),))

            pvec /= np.sum(pvec)
            neighbors = list(self.world.neighbors(n)) + [n]
            targets = Counter(np.random.choice(neighbors, int(idle_cars), p=pvec))
            neighbor_index = {}
            for i in range(len(neighbors)):
                neighbor_index[neighbors[i]] = output_index[i]
            for t in targets:
                if t != n:
                    action_tuples.append((n,t,targets[t]))
                output_multiplicity[neighbor_index[t]] += 1
        return action_tuples, output, output_multiplicity

    def train(self, db_save_callback = None):
        logging.info("Training Value Function...")
        self.valueFunc = ValIterSolver(**self.params)
        self.valueFunc.verbose = self.verbose
        self.valueFunc.train()
        logging.info("Training policy...")
        self.do_gradient()

    def get_state_value(self, time_period, world):
        # let value be a sum of values of all cells multiplied by number of cars there
        return np.sum([self.valueFunc.get_node_value(time_period, n)*world.nodes[n]['o'].get_driver_numbers() for n in world.nodes()])

    def do_gradient(self):
        '''
        state by stateprocessor is total cars and customers,
        but context is the non-negative difference, i.e. the prediction of leftovers
        '''

        t1 = time.time()
        time_periods = self.params['dataset']["time_periods"]
        rewards = []

        total_train_days = self.first_test_day
        if self.verbose:
            pbar = tqdm(total=total_train_days, desc="Training policy gradient")
        for day in np.arange(total_train_days):
            curr_state = self.env.reset_clean(city_time=day*self.time_periods) # [[cars, customers]]
            info = self.env.step_pre_order_assigin(curr_state)
            context = stateProcessor.compute_context(info) # context - [remain_drivers, remain_customers]
            curr_s = stateProcessor.utility_conver_states(curr_state) # [cars customers] flattened
            normalized_curr_s = stateProcessor.utility_normalize_states(curr_s, len(self.world)) # normalized by max_driver + max_order
            world_state = self.get_linear_state(self.env.world, normalized_curr_s, context)

            advantages = []
            total_reward = 0
            for ii in np.arange(time_periods):
                predicted_current = self.get_state_value(ii, self.env.world)
                action_tuple, output_action_vector, output_multiplicity = self.action(world_state)
                next_state, reward, info = self.env.step(action_tuple)
                total_reward += reward

                curr_state = next_state
                curr_s = stateProcessor.utility_conver_states(next_state)
                normalized_curr_s = stateProcessor.utility_normalize_states(curr_s, len(self.world))
                context = stateProcessor.compute_context(info[1])
                world_state = self.get_linear_state(self.env.world, normalized_curr_s, context)

                if ii == time_periods - 1:
                    discounted_predicted_future = 0
                else:
                    discounted_predicted_future = 0.9*self.get_state_value(ii+1, self.env.world)
                # action in advantage is a real action along sampled trajectory, so reward is a simple cumulative reward from environment
                advantage = discounted_predicted_future + reward - predicted_current
                # print(advantage, discounted_predicted_future , reward , predicted_current)
                advantages.append((advantage, world_state, output_action_vector, output_multiplicity))

            self.update_weights(advantages)
            rewards.append(total_reward)
            #TODO check conflict drivers - a very good idea! conflict drivers and average revenue among drivers as input too...

            if self.verbose:
                pbar.update()

        if self.verbose:
            pbar.close()

        self.log['train_time'] = time.time() - t1
        self.log['train_rewards'] = rewards

    def update_weights(self, advantages):
        '''
        update weights along trajectory

        advantages are scalars that contain rewards of environment (a sum over all cells, fully-cooperative)

        pi(a) = prod a^(cars)
        log pi(a) = sum_action #cars * log(sum(theta_{to_s}*s))
        theta_{to_s} is coefs from all input state variables to a specific action (action is source/dest pair)
        we take d theta - deriv by all coefs, so in the action sum all terms that are not from a specific source node are zero
        d log pi(a) = sum #cars * 1/sum(theta_{to_s}*s) * d(sum(theta_to_s * s))/d theta = < sum_action #cars * state /action for all theta_s>
        so at the end we have a matrix of theta_shape, where each element is < sum_actions #cars * state/action_probability >
        where state is the state value that corresponds to that theta, meaning the transition from a particular state to particular action

        theta matrix is 2D: < state_size , output_action_size >
        so for example, the change to [0,1] = #cars(to that output) * state[0] / output_vector[1]
        '''
        # print("Weight updates")
        for advantage, world_state, output_action_vector, number_of_cars_per_action in advantages:
            a = number_of_cars_per_action / output_action_vector
            a = np.nan_to_num(a)
            delta = np.outer(world_state, a)
            self.weights += self.alpha*delta*advantage

    def get_dispatch_action(self, env, state, context):
        curr_s_flattened = stateProcessor.utility_conver_states(state)
        normalized_curr_s = stateProcessor.utility_normalize_states(curr_s_flattened, len(env.world)) # normalized by max_driver + max_order
        context_flatten = stateProcessor.compute_context(context)
        world_state = self.get_linear_state(env.world, normalized_curr_s, context_flatten)
        action_tuple, _, _ = self.action(world_state)
        return action_tuple

    def load(self):
        pass # refactor eventually

    def save(self):
        pass # refactor eventually
