import configparser
import logging
import pickle as pkl
import sys
import os
import numpy as np
import networkx as nx
import json
from collections import Iterable
import hashlib
import pandas as pd
import tensorflow as tf
from google.protobuf.json_format import MessageToJson

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_config(config_name):
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..', config_name + ".json")
    f = open(filename)
    config = json.load(f)
    f.close()
    return config

def mov_avg(arr):
    if len(arr) < 2:
        return arr
    return np.divide(np.cumsum(arr), np.array(range(1,len(arr)+1)))


def el_in_set(set1, set2):
    '''
    Check if any element from set1 is in set2
    '''
    for el in set1:
        if el in set2:
            return True
    return False


def set_in_set(set1, set2):
    '''
    Check if all elements of set1 are in set2
    '''
    for el in set1:
        if el not in set2:
            return False
    return True


def flatten(l):
    return [item for sublist in l for item in sublist]


def flip_result_dict(data_dict):
    result = {}
    for solname in data_dict:
        for paramset in data_dict[solname]:
            if paramset not in result:
                result[paramset] = dict()
            result[paramset][solname] = data_dict[solname][paramset]
    return result


def get_static_hash(string):
    h = int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)
    return h


def load_graph(f):
    ff = open(f,'rb')
    a = pkl.load(ff)
    ff.close()
    return a


def load_seeds(f):
    return np.atleast_1d(np.loadtxt(f))


def transform_point(point, lat0, lon0):
    lat = point[1]
    lon = point[0]
    dlat = lat - lat0
    dlon = lon - lon0
    latitudeCircumference = 40075160. * math.cos(lat0 * math.pi/180.0)
    resX = dlon * latitudeCircumference / 360. 
    resY = dlat * 40008000. / 360. 
    return [resX, resY]

def transform_coords(polygon):
    '''
    Transforms all points in shapely polygons
    '''
    new_polygon = []
    for point in polygon:
        new_polygon.append(transform_point(point))
    return new_polygon

def is_solver_correct(dir_name, solver_name, solver_params):
    if dir_name.find("_test") > 0:
        return False
    vals_in_dir = set(dir_name.split("_"))
    if solver_name not in vals_in_dir:
        return False
    for s, v in solver_params:
        if isinstance(v, bool):
            val = str(int(v))
        else:
            val = "%.4f" % v if isinstance(v, float) else str(v)
        k = "".join([word[:3].capitalize() for word in s.split("_")])
        if k + val not in vals_in_dir:
            return False
    return True

def load_tb_summary_as_df(experiment_name, plotting_param, solver_name, solver_key_params=[], smoothing=0):
    """
    :param plotting_param: a str that must be contained in the name of the loaded param


    Finds and loads as Pandas dataframe results of the solver with the name and key params.
    Key params are a list of < key , value > pairs that uniqly indicate the solver instance.
    If multiple runs are found for the key_params, then they all are loaded, averaged, and std is added.
    """
    data_dir = os.path.join(os.environ['ALLDATA_PATH'], "generated", experiment_name)
    assert len(os.listdir(data_dir)) == 1, "There must be exactly one generated network per experiment. Other versions not supported"
    data_dir = os.path.join(data_dir, os.listdir(data_dir)[0])

    # traverse through all dirs, and check for each dir if it corresponds to the solver name and params
    dirs = [d for d in os.listdir(data_dir) if not os.path.isfile(os.path.join(data_dir, d)) \
                    and is_solver_correct(d, solver_name, solver_key_params)]
    if len(dirs) == 0:
        raise Exception("There is no such solver name with such params")
    if len(dirs) > 1:
        raise Exception("There are several solvers with such params")

    data_dir = os.path.join(data_dir, dirs[0])
    list_for_df = []
    # There might be several runs for the same solver
    for run_id in os.listdir(data_dir):
        path = os.path.join(data_dir, run_id, "_1")
        if len(os.listdir(path)) != 1:
            raise Exception("There are more than one run for the same footprint, \
                            check that get_footprint_params of the solver has all the variable params")
        first_file = next(os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
        for summary in tf.train.summary_iterator(first_file):
            step = summary.step
            serialized = json.loads(MessageToJson(summary.summary))
            try:
                for d in serialized['value']:
                    if (d['tag'].find(plotting_param) > -1) and (d['tag'].find(plotting_param + "_") == -1):
                        list_for_df.append({"step": step, "val": d['simpleValue'], "run_id": run_id})
                        # this might raise KeyError if the requested params is a hist (an array). Then there is no simpleValue.
            except KeyError:
                pass
    df = pd.DataFrame(list_for_df)
    if df.duplicated().any():
        raise Exception("There are duplicated values, check uniquness of the plotting parameter")
    if len(df) == 0:
        raise Exception("No results, empty dataframe")

    # first do smoothing, then averaging!
    win_size = int(0.1 * len(df))
    for run_id in df["run_id"].unique():
        s = df[df["run_id"] == run_id].rolling(win_size).sum()['val']
        df.loc[df["run_id"] == run_id, 'val'] = s
    df = df[df['val'].notnull()]
    
    df = df.groupby(by="step", as_index=False).agg(['mean', 'std'])
    df.reset_index(inplace=True)
    return df