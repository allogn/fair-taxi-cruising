from stable_baselines import PPO2
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'src', 'Expert'))
from FileManager import *

fm = FileManager("cartpole_test")
fm.clean_data_path()
log_path = fm.get_data_path()

model = PPO2('MlpPolicy', 'CartPole-v1', tensorboard_log=log_path, verbose=1)
model.learn(10000)
