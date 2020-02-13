import os, sys
import pickle as pkl
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from OrigSolver import *

from o_baseline_rulebased_valueiter import *

class OrigValIterSolver(OrigSolver):
    def __init__(self, **params):
        super().__init__(**params)
        self.dpath = self.params['dataset']["dataset_path"]

    def test(self):
        pass

    @staticmethod
    def compute_context(target_grids, info):

        context = info.flatten()
        context = [context[idx] for idx in target_grids]
        return context

    def train(self):
        n_side = 6
        GAMMA = 0.9
        l_max = 9

        self.env = CityReal(self.mapped_matrix_int, self.order_num_dist,
                       self.idle_driver_dist_time, self.idle_driver_location_mat,
                       self.order_time_dist, self.order_price_dist,
                       self.l_max, self.M, self.N, self.n_side, self.order_sample_p, self.order_real, self.onoff_driver_location_mat,
                       global_flag="")

        temp = np.array(self.env.target_grids) + self.env.M * self.env.N
        target_id_states = self.env.target_grids + temp.tolist()

        RATIO = 1 # orig 0.1
        # value_table_ave = pickle.load(open("value_table_ave", 'rb')) @alvis --- probably rule-based is used as initialization, not sure
        value_table_ave = np.zeros((144, self.N*self.M))

        MAX_ITER = 50  # 10 iteration the Q-learning loss will converge.
        is_plot_figure = False
        city_time_start = 0
        EP_LEN = 144
        global_step = 0
        city_time_end = city_time_start + EP_LEN
        epsilon = 0.5
        gamma = 0.9
        learning_rate = 1e-3

        prev_epsiode_reward = 0
        curr_num_actions = []
        all_rewards = []
        order_response_rate_episode = []
        value_table_sum = []
        episode_rewards = []
        num_conflicts_drivers = []
        episode_conflicts_drivers = []
        episode_dispatched_drivers = []

        T = 144
        action_dim = 7
        state_dim = self.env.n_valid_grids * 3 + T

        record_all_order_response_rate = []

        model = ValueIter(value_table_ave, self.env, 0.9, 0.9)

        prev_values = np.sum(model.value_table)
        save_random_seed = []
        for n_iter in np.arange(25):
            RANDOM_SEED = n_iter + MAX_ITER - 10
            self.env.reset_randomseed(RANDOM_SEED)
            save_random_seed.append(RANDOM_SEED)
            batch_s, batch_a, batch_r = [], [], []
            batch_reward_gmv = []
            epsiode_reward = 0
            num_dispatched_drivers = 0

            is_regenerate_order = 0
            curr_state = self.env.reset_clean(generate_order=is_regenerate_order, ratio=RATIO, city_time=city_time_start)

            curr_s = np.array(curr_state).flatten()
            curr_s = utility_conver_states(curr_s, target_id_states)

            context = self.env.step_pre_order_assigin(curr_state)[0].flatten()
            context = [context[idx] for idx in self.env.target_grids]

            # record rewards to update the value table
            episodes_immediate_rewards = []
            order_response_rates = []
            curr_num_actions = []
            num_conflicts_drivers = []
            for ii in np.arange(EP_LEN + 1):
                devide = curr_s[:self.env.n_valid_grids]
                devide[devide == 0] = 1

                action_tuple = model.action(context, self.env.city_time % 144)

                # ONE STEP: r0
                next_state, r, info = self.env.step(action_tuple, 2)

                info_reward = info[0]
                context = info[1][0].flatten()
                context = [context[idx] for idx in self.env.target_grids]
                order_response_rates.append(self.env.order_response_rate)

                immediate_reward = utility_conver_reward(info_reward[0], self.env.target_grids)

                curr_state = next_state
                curr_s = np.array(next_state).flatten()
                curr_s = utility_conver_states(curr_s, target_id_states)

                immediate_reward = immediate_reward / devide  # 到达该格子的司机的平均reward.
                episodes_immediate_rewards.append(immediate_reward)

                # book keeping
                global_step += 1
                all_rewards.append(r)
                batch_reward_gmv.append(r)
                order_response_rates.append(self.env.order_response_rate)
                num_conflicts_drivers.append(collision_action(action_tuple))
                curr_num_action = np.sum([aa[2] for aa in action_tuple]) if len(action_tuple) != 0 else 0
                curr_num_actions.append(curr_num_action)

            episodes_immediate_rewards = episodes_immediate_rewards[1:]
            for jj in np.arange(EP_LEN)[::-1]:
                jj_rewards = episodes_immediate_rewards[jj]  # time jj=143, 142, 141, ...,0
                model.value_iterate_updates(jj_rewards, jj+1)  # reward r=144, 143, ...1

            episode_reward = np.sum(batch_reward_gmv[1:])
            episode_rewards.append(episode_reward)
            n_iter_order_response_rate = np.mean(order_response_rates[1:])
            order_response_rate_episode.append(n_iter_order_response_rate)
            record_all_order_response_rate.append(order_response_rates)
            episode_conflicts_drivers.append(np.sum(num_conflicts_drivers[:-1]))
            episode_dispatched_drivers.append(np.sum(curr_num_actions[:-1]))
            logging.info("iteration {} ********* reward {} order{} conflicts {} drivers {} seed{}".format(n_iter, episode_reward,
                                                                                            order_response_rate_episode[-1],
                                                                                            episode_conflicts_drivers[-1],
                                                                                            episode_dispatched_drivers[-1], RANDOM_SEED))


            curr_values = np.sum(model.value_table)

            value_table_sum.append(np.abs(curr_values - prev_values))
            prev_values = curr_values

            with open(os.path.join(self.dpath, "results.pkl"), "wb") as f:
                pkl.dump([episode_rewards, order_response_rate_episode, save_random_seed, value_table_sum,
                         episode_conflicts_drivers,
                         episode_dispatched_drivers], f)

    def load(self):
        pass # refactor eventually

    def save(self):
        pass # refactor eventually
