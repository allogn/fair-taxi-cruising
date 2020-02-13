import gym
import imageio
import os
import numpy as np
import shutil

from stable_baselines import A2C
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import set_global_seeds

gym.envs.register(
    id='TaxiDummy-v01',
    entry_point='gym_taxi.envs:TaxiEnvDummy',
    kwargs={'gridsize': 10}
)

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


if __name__ == '__main__':
    env_id = "TaxiDummy-v01"

    DATA_PATH = os.path.join(os.environ['ALLDATA_PATH'], "macaoFiles", "taxi_env_dummy")
    if os.path.isdir(DATA_PATH):
        shutil.rmtree(DATA_PATH)
    os.makedirs(DATA_PATH)

    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    model = A2C(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=100000)

    obs = env.reset()
    images = []
    img = env.render(mode="rgb_array")
    images.append(img)
    for _ in range(70):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        images.append(env.render(mode="rgb_array"))
    imageio.mimwrite(os.path.join(DATA_PATH, 'taxi_dummy_a2c.gif'), [np.array(img) for i, img in enumerate(images)], format="GIF-PIL", fps=5)
