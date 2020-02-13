import os, sys
import pickle as pkl
from src.Solvers.Solver import Solver
from src.Generator import Generator
import gym
from tqdm import tqdm
import time
import uuid
import numpy as np
import logging
import imageio

class TestingSolver(Solver):
    def __init__(self, **params):
        super().__init__(**params)
        self.time_periods = self.params['dataset']["time_periods"]
        self.days = self.params['dataset']["days"]
        self.load_dataset()
        self.init_gym()

    def init_gym(self):
        env_params = {
            "world": self.world,
            "orders": self.real_orders,
            "order_sampling_rate": 1./self.params['dataset']['days']*self.params['dataset']['order_sampling_multiplier'],
            "drivers_per_node": self.idle_driver_locations[0,:],
            "n_intervals": self.time_periods,
            "wc": self.params["wc"],
            "count_neighbors": self.params['count_neighbors'] == 1,
            "weight_poorest": 0,
            "normalize_rewards": 0,
            "minimum_reward": 0,
            "include_income_to_observation": self.params.get('include_income_to_observation', 0) == 1,
            "poorest_first": self.params.get("poorest_first", 0) == 1,
            "idle_reward": self.params.get("idle_reward", 0) == 1
        }
        env_id = "TaxiEnvBatch{}-v01".format(str(uuid.uuid4()))
        gym.envs.register(
            id=env_id,
            entry_point='gym_taxi.envs:TaxiEnvBatch',
            kwargs=env_params
        )
        self.testing_env = gym.make(env_id)

    def load_dataset(self):
        '''
        load complete dataset
        note that orders are merged into a single day, and then sampled out of there
        '''
        dataset_params = self.params['dataset']
        gen = Generator(self.params['tag'], dataset_params)
        self.world, self.idle_driver_locations, self.real_orders, \
            self.onoff_driver_locations, random_average, dist = gen.load_complete_set(dataset_id=self.params['dataset']['dataset_id'])

    def do_test_iteration(self, draw=False):
        stats = {}
        t = time.time()
        randseed = np.random.randint(1,100000)
        stats['seed'] = randseed
        self.testing_env.seed(randseed)
        state = self.testing_env.reset()
        info = self.testing_env.get_reset_info()
        rewards = []
        min_income = []
        idle_reward = []
        min_idle = []
        done = False
        it = 0
        order_response_rates = []
        nodes_with_drivers = []
        nodes_with_orders = []

        images = []

        while not done:
            action = self.predict(state, info)
            state, reward, done, info = self.testing_env.step(action)

            if draw and it == 1:
                images.append(self.testing_env.render(mode="rgb_array"))

            order_response_rates.append(float(info['served_orders']/(info['total_orders']+0.0001)))
            nodes_with_drivers.append(int(info['nodes_with_drivers']))
            nodes_with_orders.append(int(info['nodes_with_orders']))
            min_income.append(self.testing_env.get_min_revenue())
            rewards.append(reward)
            idle_reward.append(info['idle_reward'])
            min_idle.append(info['min_idle'])
            it += 1
        assert it == self.time_periods, (it, self.time_periods)
        stats['income_distr'] = [float(d.get_income()) for d in self.testing_env.itEnv.all_driver_list]
        stats['order_response_rates'] = float(np.mean(order_response_rates))
        stats['order_response_rates_std'] = float(np.std(order_response_rates))
        stats['nodes_with_drivers'] = float(np.mean(nodes_with_drivers))
        stats['nodes_with_orders'] = float(np.mean(nodes_with_orders))
        stats['nodes_with_drivers_std'] = float(np.std(nodes_with_drivers))
        stats['nodes_with_orders_std'] = float(np.std(nodes_with_orders))
        stats['min_income'] = float(min_income[-1])
        stats['rewards'] = float(np.sum(rewards))
        stats['min_idle'] = float(min_idle[-1])
        stats['idle_reward'] = float(np.mean(idle_reward))
        stats['testing_iteration_time'] = time.time() - t
        return stats, images

    def test(self):
        t1 = time.time()
        self.log['seeds'] = []
        total_reward_per_epoch = []
        total_min_reward_per_epoch = []
        total_min_idle_per_epoch = []
        total_idle_per_epoch = []

        total_test_days = self.params['testing_epochs']

        if self.verbose:
            pbar = tqdm(total=total_test_days, desc="Testing Solver")
        for day in range(total_test_days):
            stats, images = self.do_test_iteration(draw = False)
            # need to rereun all experiments in server to plot because current ones
            # are done with graph with missing coordinates

            total_min_reward_per_epoch.append(stats['min_income'])
            total_reward_per_epoch.append(np.sum(stats['rewards']))
            total_min_idle_per_epoch.append(stats['min_idle'])
            total_idle_per_epoch.append(np.mean(stats['idle_reward']))
            self.log.update(stats)
            if self.verbose:
                pbar.update()

        if self.verbose:
            pbar.close()

        self.log['test_total_min_reward_per_epoch'] = float(np.mean(total_min_reward_per_epoch))
        self.log['test_total_reward_per_epoch'] = float(np.mean(total_reward_per_epoch))
        self.log['test_total_min_reward_per_epoch_std'] = float(np.std(total_min_reward_per_epoch))
        self.log['test_total_reward_per_epoch_std'] = float(np.std(total_reward_per_epoch))

        self.log['test_total_min_idle_per_epoch_std'] = float(np.std(total_min_idle_per_epoch))
        self.log['test_total_idle_per_epoch_std'] = float(np.std(total_idle_per_epoch))
        self.log['test_total_min_idle_per_epoch'] = float(np.mean(total_min_idle_per_epoch))
        self.log['test_total_idle_per_epoch'] = float(np.mean(total_idle_per_epoch))

        self.log['test_test_time'] = time.time() - t1

        logging.info("Testing finished with total obj {}, min obj {}".format(self.log['test_total_reward_per_epoch'], self.log['test_total_min_reward_per_epoch']))

        if len(images) > 0:
            imageio.mimwrite(os.path.join(self.dpath, 'taxi_env.gif'),
                                [np.array(img) for i, img in enumerate(images)], format="GIF-PIL", fps=5)

    def predict(self, state, info):
        raise NotImplementedError()
