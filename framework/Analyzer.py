import pandas as pd
import numpy as np
import sys
import logging
import os
import json
import warnings

import framework.helpers as helpers
from framework.FileManager import *
from framework.Artist import *
from framework.ParameterManager import *
from framework.MongoDatabase import *

class Analyzer:
    def __init__(self, tag):
        self.tag = tag
        self.df = pd.DataFrame()
        self.db_wrapper = MongoDatabase(tag)
        self.db = self.db_wrapper.db
        self.load_dataframe()

    def load_dataframe(self):
        q = {"tag": self.tag, "mode": "Test"}
        for solver_result in self.db.solution.find(q):
            row = solver_result

            if 'dataset' in row:
                dataset_params = row['dataset']
                row.update(dataset_params)
                del row['dataset']

            self.df = self.df.append(row,ignore_index=True)
        if len(self.df) == 0:
            raise Exception("No results for the tag")
