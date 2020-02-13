# for macOS run this first: export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

import numpy as np
import torch
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'lib', 'ppo-pytorch'))

from agents import PPO
from curiosity import NoCuriosity
from envs import MultiEnv
from models import MLP
from reporters import TensorBoardReporter
from rewards import GeneralizedAdvantageEstimation, GeneralizedRewardEstimation
import gym

gym.envs.register(
    id='TaxiDummy-v01',
    entry_point='gym_taxi.envs:TaxiEnvDummy',
    kwargs={'gridsize': 10}
)

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reporter = TensorBoardReporter()

    agent = PPO(MultiEnv('TaxiDummy-v01', 4, reporter),
                reporter=reporter,
                normalize_state=False,
                normalize_reward=False,
                model_factory=MLP.factory(),
                curiosity_factory=NoCuriosity.factory(),
                reward=GeneralizedRewardEstimation(gamma=0.99, lam=0.95),
                advantage=GeneralizedAdvantageEstimation(gamma=0.99, lam=0.95),
                learning_rate=5e-3,
                clip_range=0.2,
                v_clip_range=0.3,
                c_entropy=1e-2,
                c_value=0.5,
                n_mini_batches=4,
                n_optimization_epochs=5,
                clip_grad_norm=0.5)
    agent.to(device, torch.float32, np.float32)

    agent.learn(epochs=200, n_steps=500)
    agent.eval(n_steps=500, render=True)
