import os, sys
import pickle as pkl
from framework.solvers.Solver import Solver
from framework.Generator import Generator
import networkx as nx
from framework.FileManager import FileManager
from framework.Artist import Artist
import gym
from tqdm import tqdm
import time
import uuid
import numpy as np
import logging
import tensorflow as tf
from collections.abc import Iterable

class TestingSolver(Solver):
    def __init__(self, **params):
        super().__init__(**params)
        self.time_periods = self.params['dataset']["time_periods"]
        self.days = self.params['dataset']["days"]
        self.load_dataset()
        self.init_gym(self.params.get('testing_seed',0))

        # tf logging
        # self.artist = Artist()
        self.fm = FileManager(self.params['tag'])
        self.log_dir = os.path.join(self.dpath,self.get_solver_signature())
        self.fm.create_path(self.log_dir)
        # tf.reset_default_graph()  # important! logging works weirdly otherwise, creates separate plots per iteration
        # # also important to reset before session, not after

        # self.sess = tf.Session()
        self.test_tf_writer = tf.summary.FileWriter(self.log_dir)
        # self.epoch_stats = {}
        # self.summaries = None
        

    def init_gym(self, testing_seed):
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
            "idle_reward": self.params.get("idle_reward", 0) == 1,
            "seed": testing_seed
        }
        env_id = "TaxiEnvBatch{}-v01".format(str(uuid.uuid4()))
        gym.envs.register(
            id=env_id,
            entry_point='gym_taxi.envs:TaxiEnvBatch',
            kwargs=env_params
        )
        self.test_env = gym.make(env_id)

    def load_dataset(self):
        '''
        load complete dataset
        note that orders are merged into a single day, and then sampled out of there
        '''
        dataset_params = self.params['dataset']
        gen = Generator(self.params['tag'], dataset_params)
        self.world, self.idle_driver_locations, self.real_orders, \
            self.onoff_driver_locations, random_average, dist = gen.load_complete_set(dataset_id=self.params['dataset']['dataset_id'])

    def run_test_episode(self, training_iter, draw=False, debug=True):
        stats = {}
        t = time.time()
        randseed = np.random.randint(1,100000)
        stats['seed'] = float(randseed)
        self.test_env.seed(randseed)
        self.test_env.DEBUG = debug
        state = self.test_env.reset()
        info = self.test_env.get_reset_info()
        done = False

        i = 0
        while not done:
            action = self.predict(state, info)
            state, reward, done, info = self.test_env.step(action)
            if draw:
                fig = self.test_env.render('fig')
                fig_dir = os.path.join(self.log_dir, str(training_iter))
                self.fm.create_path(fig_dir)
                fig.savefig(os.path.join(fig_dir, str(i) + "_fig.png"), dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)
            i += 1
        
        stats.update(self.test_env.get_episode_info())
        stats['test_episode_runtime'] = time.time() - t
        return stats

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

    def test(self):
        raise NotImplementedError("Testing is performed during training for all solvers. Testing on unobserved data is not implemented.")
        # self.run_tests()

    def run_tests(self, training_iteration, draw = False, verbose = 1):
        t1 = time.time()
        self.log['seeds'] = []

        # average per testing iteration
        test_stats = {
            "total_min_income_per_epoch": [],
            "total_sum_income_per_epoch": [],
            "total_avg_income_per_epoch": [],
            "total_reward_per_epoch": [],
            "total_min_idle_per_epoch": [],
            "total_avg_idle_per_epoch": [],
            "total_sum_idle_per_epoch": []
        }

        total_test_days = self.params['testing_epochs']

        if verbose:
            pbar = tqdm(total=total_test_days, desc="Testing Solver")

        for day in range(total_test_days): # number of episodes
            # plot and check consistency only at first iteration to save time and space
            stats = self.run_test_episode(training_iteration, draw and day == 0, day == 0) 
            # need to rereun all experiments in server to plot because current ones
            # are done with graph with missing coordinates

            test_stats["total_min_income_per_epoch"].append(np.min(stats['driver_income']))
            test_stats["total_sum_income_per_epoch"].append(np.sum(stats['driver_income']))
            test_stats["total_avg_income_per_epoch"].append(np.sum(stats['driver_income']))
            test_stats["total_reward_per_epoch"].append(np.sum(stats['rewards']))
            test_stats["total_min_idle_per_epoch"].append(np.min(stats['driver_income']))
            test_stats["total_avg_idle_per_epoch"].append(np.mean(stats['idle_periods']))
            test_stats["total_sum_idle_per_epoch"].append(np.sum(stats['idle_periods']))
            if verbose:
                pbar.update()

        if verbose:
            pbar.close()

        values = []
        for k, val in test_stats.items():
            values.append(tf.Summary.Value(tag="test_stats/" + k + '_mean', simple_value=float(np.mean(val))))
            values.append(tf.Summary.Value(tag="test_stats/" + k + '_std', simple_value=float(np.std(val))))
        summary = tf.Summary(value=values)
        self.test_tf_writer.add_summary(summary, training_iteration)
        self.log['test_test_time'] = time.time() - t1

    def predict(self, state, info):
        raise NotImplementedError()
