import os, sys
import pickle as pkl
from framework.solvers.Solver import Solver
from framework.Generator import Generator
from framework.FileManager import FileManager
from framework.Artist import Artist
import gym
from tqdm import tqdm
import time
import uuid
import numpy as np
import logging
import imageio
import io
import tensorflow as tf
from collections.abc import Iterable
import matplotlib.pyplot as plt

class TestingSolver(Solver):
    def __init__(self, **params):
        super().__init__(**params)
        self.time_periods = self.params['dataset']["time_periods"]
        self.days = self.params['dataset']["days"]
        self.load_dataset()
        self.init_gym()

        # tf logging
        self.artist = Artist()
        self.fm = FileManager(self.params['tag'])
        train_log_dir = os.path.join(self.fm.get_data_path(),self.get_solver_signature())
        tf.reset_default_graph()  # important! logging works weirdly otherwise, creates separate plots per iteration
        # also important to reset before session, not after

        self.sess = tf.Session()
        self.summary_writer = tf.summary.FileWriter(train_log_dir)
        self.epoch_stats = {}
        self.summaries = None
        

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

    def run_test_episode(self, draw=False):
        stats = {}
        t = time.time()
        randseed = np.random.randint(1,100000)
        stats['seed'] = float(randseed)
        self.testing_env.seed(randseed)
        state = self.testing_env.reset()
        info = self.testing_env.get_reset_info()
        rewards = []
        min_income = []
        total_income = []
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

            if draw:
                images.append(self.testing_env.render())

            order_response_rates.append(float(info['served_orders']/(info['total_orders']+0.0001)))
            nodes_with_drivers.append(int(info['nodes_with_drivers']))
            nodes_with_orders.append(int(info['nodes_with_orders']))
            min_income.append(self.testing_env.get_min_revenue())
            rewards.append(reward)
            idle_reward.append(info['idle_reward'])
            min_idle.append(float(info['min_idle']))
            it += 1
        
        # take only subset of images
        # we don't know how many of them in total, so we render all
        images = [images[i] for i in range(0,len(images),len(images)//5)]
        
        # env can go through several time steps per iteration, not no more than n_interations
        assert it <= self.time_periods, (it, self.time_periods)
        
        stats['order_response_rates'] = order_response_rates
        stats['nodes_with_drivers'] = nodes_with_drivers
        stats['nodes_with_orders'] = nodes_with_orders
        stats['min_income'] = float(min_income[-1])
        stats['rewards'] = rewards
        stats['min_idle'] = min_idle
        stats['idle_reward'] = float(np.mean(idle_reward))
        stats['testing_iteration_time'] = float(time.time() - t)

        figure = self.artist.combine_drawings(images)

        return stats, figure

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image


    def save_tf_summary(self, episode, stats, figure):
        assert len(stats) > 0 
        for k, val in stats.items():
            if isinstance(val, Iterable):
                self.epoch_stats[k] = val
                self.epoch_stats[k + "_mean"] = float(np.mean(val))
                self.epoch_stats[k + "_std"] = float(np.std(val))
            else:
                assert type(val) == float
                assert type(k) == str
                self.epoch_stats[k] = val
            self.epoch_stats["visual"] = self.plot_to_image(figure)
        with tf.name_scope('stats'):
            if self.summaries is None:
                summaries = []
                for k, val in self.epoch_stats.items():
                    if k == 'visual':
                        summaries.append(tf.summary.image(k, self.epoch_stats[k]))
                        continue
                    if isinstance(val, Iterable):
                        summaries.append(tf.summary.histogram(k, self.epoch_stats[k]))
                    else:
                        assert type(val) == float
                        assert type(k) == str
                        summaries.append(tf.summary.scalar(k, self.epoch_stats[k]))
                self.summaries = tf.summary.merge(summaries)
            
            summary = self.sess.run(self.summaries)
            self.summary_writer.add_summary(summary, episode)
            self.summary_writer.flush()

    def test(self):
        self.run_tests() # some solvers run tests during training stage

    def run_tests(self, draw = False):
        t1 = time.time()
        self.log['seeds'] = []
        total_reward_per_epoch = []
        total_min_reward_per_epoch = []
        total_min_idle_per_epoch = []
        total_idle_per_epoch = []

        total_test_days = self.params['testing_epochs']

        if self.verbose:
            pbar = tqdm(total=total_test_days, desc="Testing Solver")

        for day in range(total_test_days): # number of episodes
            stats, figure = self.run_test_episode(draw)
            self.save_tf_summary(day, stats, figure)
            # need to rereun all experiments in server to plot because current ones
            # are done with graph with missing coordinates

            total_min_reward_per_epoch.append(stats['min_income'])
            total_reward_per_epoch.append(np.sum(stats['rewards']))
            total_min_idle_per_epoch.append(stats['min_idle'][-1])
            total_idle_per_epoch.append(np.mean(stats['idle_reward']))
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

        # if len(images) > 0:
        #     imageio.mimwrite(os.path.join(self.dpath, 'taxi_env.gif'),
        #                         [np.array(img) for i, img in enumerate(images)], format="GIF-PIL", fps=5)

    def predict(self, state, info):
        raise NotImplementedError()
