import time
import json
import uuid

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
        pass

    def get_name(self):
        return self.__class__.__name__

    def run(self):
        if "mode" not in self.params:
            self.train()
            self.test()
            return self.log
        if self.params["mode"] == "Train":
            self.train()
            self.save()
        if self.params["mode"] == "Test":
            self.load()
            self.test()
        return self.log

    def train(self):
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
