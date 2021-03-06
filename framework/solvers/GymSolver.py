import os, sys
import imageio
import os
import numpy as np
import shutil
import gym
import time
import logging
import tensorflow as tf

from stable_baselines import A2C, PPO2, ACKTR
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import CheckpointCallback

from framework.solvers.TestingSolver import TestingSolver
from framework.Generator import Generator
from framework.FileManager import FileManager
from framework.solvers.cA2C.oenvs import CityReal
from framework.ParameterManager import ParameterManager
from framework.solvers.callbacks import TensorboardCallback, TestingCallback, RobustCallback

class GymSolver(TestingSolver):
    def __init__(self, **params):
        super().__init__(**params)
        self.Model = PPO2
        num_cpu = self.params['num_cpu']

        # parameters from our config, not the original one
        self.days = self.params['dataset']["days"]
        env_id = "TaxiEnvBatch-v01"
        self.env_params = self.load_env_params()

        seed = self.params['seed']
        # Create the vectorized environment
        self.train_env = SubprocVecEnv([self.make_env(env_id, i, seed+i, self.views, self.env_params) for i in range(num_cpu)])
        self.train_env = VecNormalize(self.train_env, norm_obs=True, norm_reward=True)

        # testing not implemented so far
        # self.test_env_native = SubprocVecEnv([self.make_env(env_id, 1, seed+num_cpu+1, self.env_params)])
        # self.test_env_native = VecNormalize(self.test_env_native, norm_obs=False, norm_reward=False)

    def init_model(self):
        if self.params.get("lstm", 0) == 1:
            Policy = MlpLstmPolicy
            nminibatches = 1
            num_cpu = 1 # One current limitation of recurrent policies is that you must test them with the same number of environments they have been trained on.
        else:
            Policy = MlpPolicy
            nminibatches = 1 # 16
            num_cpu = self.params['num_cpu']

        policy_params=[0, dict(pi=[128, 64, 32], vf=[128, 64, 32])] # 0 - shared layers
        seed = self.params['seed']
        self.log_dir = os.path.join(self.log_dir)

        n_steps = (self.params['dataset']['time_periods']+1)
        # n_steps is time_periods + 1 because we have the cold-start, which is a fake iteration.
        if not self.params['discrete'] or not self.params['batch_env']:
            n_steps = n_steps*self.params['dataset']['number_of_cars']

        # policy_params = [128, 64, 32]
        self.model = self.Model(Policy, self.train_env,
                                gamma=self.params['gamma'], ent_coef=self.params['ent_coef'],
                                learning_rate=self.params['learning_rate'], vf_coef=self.params['vf_coef'],
                                max_grad_norm=self.params['max_grad_norm'], lam=self.params['lam'],
                                noptepochs=self.params['noptepochs'], cliprange=self.params['cliprange'],
                                seed=seed, verbose=0, nminibatches=nminibatches, 
                                policy_kwargs={"net_arch": policy_params},
                                tensorboard_log=self.log_dir, full_tensorboard_log=False,
                                n_steps=n_steps)

        # number of steps might be very large, because it might have to go through all nodes

    def get_footprint_params(self):
        footprint_params = {
            "n_intervals": self.time_periods,
            "wc": self.params["wc"],
            "count_neighbors": self.params['count_neighbors'] == 1,
            "weight_poorest": self.params['weight_poorest'] == 1,
            "normalize_rewards": self.params['normalize_rewards'] == 1,
            "minimum_reward": self.params['minimum_reward'] == 1,
            "include_income_to_observation": self.params['include_income_to_observation'] == 1,
            "poorest_first": self.params.get("poorest_first", 0) == 1,
            "robust": self.params['robust'] == 1
        }
        if self.params.get("lstm", 0) == 1:
            footprint_params['lstm'] = True
        return footprint_params

    def load_env_params(self):
        '''
        load complete dataset

        note that orders are merged into a single day, and then sampled out of there
        '''
        dataset_params = self.params['dataset']
        gen = Generator(self.params['tag'], dataset_params)
        world, idle_driver_locations, real_orders, onoff_driver_locations, random_average, dist = gen.load_complete_set(dataset_id=self.params['dataset']['dataset_id'])
        params = {
            "world": world,
            "orders": real_orders,
            "order_sampling_rate": 1./self.days*self.params['dataset']['order_sampling_multiplier'],
            "drivers_per_node": idle_driver_locations[0,:],
            "n_intervals": self.params['dataset']['time_periods'],
            "wc": self.params['wc'],
            "count_neighbors": self.params['count_neighbors'] == 1,
            "weight_poorest": self.params['weight_poorest'] == 1,
            "normalize_rewards": self.params['normalize_rewards'] == 1,
            "minimum_reward": self.params['minimum_reward'] == 1,
            "include_income_to_observation": self.params['include_income_to_observation'] == 1,
            "poorest_first": self.params.get("poorest_first", 0) == 1,
            "idle_reward": self.params["idle_reward"], 
            "include_action_mask": self.params["action_mask"],
            "seed": self.params["seed"],
            "hold_observation": self.params["hold_observation"],
            "penalty_for_invalid_action": self.params["penalty_for_invalid_action"],
            "discrete": self.params["discrete"],
            "bounded_income": self.params["robust"] == 1,
            "fully_collaborative": self.params["batch_env"] == 1,
            "randomize_drivers": self.params["randomize_drivers"],
            "debug": self.params["debug"]
        }

        return params


    def make_env(self, env_id, rank, seed, views, env_params={}):
        """
        Utility function for multiprocessed env.
        Note: do not use self.<vars> in this function! _thread.lock bug appears in multiprocessing

        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environments you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        if self.params['batch_env']:
            entry_point = 'gym_taxi.envs:TaxiEnvBatch'
        else:
            entry_point = 'gym_taxi.envs:TaxiEnv'
        def _init():
            gym.envs.register(
                id=env_id,
                entry_point=entry_point,
                kwargs=env_params
            ) # must be in make_env because otherwise doesn't work
            env = gym.make(env_id)
            env.seed(seed + rank)
            if len(views) == 1:
                env.set_view(views[next(iter(views))])
            return env
        set_global_seeds(seed)
        return _init

    def get_callback(self, db_save_callback):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        def no_callback(_locals, _globals):
            return True

        if self.params.get("callback", 0) == 1:
            callbacks = [
                TensorboardCallback(),
                TestingCallback(self, verbose=0, eval_freq=self.params['eval_freq'], 
                                draw=self.params['draw'] == 1, draw_freq=self.params['draw_freq']),
                CheckpointCallback(save_freq=self.params['save_freq'], 
                                    save_path=self.log_dir, name_prefix='gymsave')]
            if self.params['robust']:
                callbacks.append(RobustCallback(self, 
                                    self.params['robust_nu'], 
                                    self.params['robust_epsilon'], 
                                    self.params['robust_gamma'],
                                    self.params['robust_cmin'], 
                                    self.params['robust_cmax'],
                                    verbose=0))
            return callbacks
        else:
            return no_callback

    def train(self, db_save_callback = None):
        # save whatever we have now, so that we can stop running at any moment without reruns
        if db_save_callback is not None:
            db_save_callback(self.log)
        t = time.time()
        self.init_model()
        self.model.learn(total_timesteps=self.params['training_iterations'], tb_log_name="",
                        callback=self.get_callback(db_save_callback))
        self.log['training_time'] = time.time() - t

    def predict(self, state, info, nn_state = None):
        if self.params["discrete"]:
            # we are using the same type of env
            action, _state = self.model.predict(state, action_mask=[info['action_mask']])
            # action is the id of neighbour 
            return action
        
        # return bunch of actions given learned state-action for singular taxi
        # state is the one returned by testing environment (multitaxi), and action returned should be for that too
        actions = []
        for n in self.world.nodes():
            onehot_nodeid = np.zeros(len(self.world))
            onehot_nodeid[n] = 1
            if self.params['include_income_to_observation'] == 1:
                assert self.test_env.include_income_to_observation
                positions_for_income = len(self.world)
                assert state[:-positions_for_income].shape == ((3*len(self.world)+self.time_periods),)
                assert state[-positions_for_income:].shape[0] == positions_for_income
                assert self.train_env.observation_space.shape == (5*len(self.world)+self.time_periods,)

                if self.params['batch_env']:
                    obs = np.concatenate((state[:-positions_for_income], state[-positions_for_income:]))
                    assert obs.shape[0] == 4*len(self.world)+self.time_periods
                else:
                    obs = np.concatenate((state[:-positions_for_income], onehot_nodeid, state[-positions_for_income:]))
                    assert obs.shape[0] == 5*len(self.world)+self.time_periods
            else:
                assert not self.test_env.include_income_to_observation
                assert state.shape == (3*len(self.world)+self.time_periods,)
                if self.params['batch_env']:
                    obs = state
                    assert obs.shape[0] == 3*len(self.world)+self.time_periods
                else:
                    obs = np.concatenate((state, onehot_nodeid))
                    assert obs.shape[0] == 4*len(self.world)+self.time_periods

            if self.params.get("lstm", 0) == 1:
                raise NotImplementedError("The problem is how to combine last states of NN from different graph nodes")
            else:
                action, _states = self.model.predict(obs)
            actions.append(action)
        actions = np.concatenate(actions)
        return actions
