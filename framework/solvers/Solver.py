import time
import json
import uuid

from framework.ParameterManager import ParameterManager

class Solver:
    def __init__(self, **params):
        self.log = params
        self.params = params
        self.verbose = False
        self.seed(1)
        self.dpath = self.params['dataset']["dataset_path"]
        self.DEBUG = params.get('debug',0) == 1

    def seed(self, seed):
        self.random_seed = seed
        
    def reset(self):
        raise NotImplementedError()

    def get_name(self):
        return self.__class__.__name__

    def get_footprint_params(self):
        return {}

    def get_solver_signature(self):
        return self.get_name()[:-6] + "_" + ParameterManager.get_param_footprint(self.get_footprint_params())

    def run(self, db_save_callback):
        if "mode" not in self.params or self.params["mode"] == "Train":
            self.train(db_save_callback)
            db_save_callback(self.log)

    def train(self, db_save_callback = None):
        raise NotImplementedError()
        return

    def test(self):
        raise NotImplementedError()
        return

    def save(self):
        raise NotImplementedError()
        return

    def load(self):
        raise NotImplementedError()
        return
