import os,sys
import networkx as nx
import random
import time
import argparse
import numpy as np
import logging
import uuid
from scipy.spatial.distance import euclidean
import scipy.stats
import pickle as pkl
from shutil import copyfile
import csv
from collections import Counter

import framework.helpers
from framework.FileManager import *

class Generator:
    def __init__(self, title, params):
        '''
        Requires params:
                - days, dataset_type, ... (type-specific, like n)

            For non-chicago:
                - time_periods, order_distr (uniform, star, centered), orders_density, number_of_cars
                    order density for star is simply number of cars. for uniform,centered - multipl. of range

        Notes:
            - Currently onoff and idle drivers are generated for one day only.
        '''


        self.params = params
        self.title = title
        self.data_path = None # should be overwritten either by loading dataset, or by generating one
        assert(title != '')
        self.fm = FileManager(title)
        self.generators = {
            'dummy': lambda: self.generate_dummy(self.params['dummy_data_param']),
            'linear': lambda: self.generate_linear(self.params['n']),
            'grid': lambda: self.generate_grid(self.params['n']),
            'hexagon': lambda: self.generate_hexagon(self.params['n']),
            'chicago': lambda: self.generate_chicago(self.params.get('sparsity', 1))
        }

        if self.params['dataset_type'] == 'chicago':
            self.params["time_periods"] = (24*60)//15

        # set default values
        self.params["time_periods_per_hour"] = self.params.get("time_periods_per_hour", self.params["time_periods"]) # static

    def gen_id(self):
        return str(uuid.uuid4())

    def load_complete_set(self, dataset_id = None):
        '''
        return: world_graph, idle_driver_locations, real_orders, onoff_driver_locations, random_average, dist

        can be used either after calling "generate", or passing dataset_id
        '''
        if dataset_id != None:
            assert self.data_path == None, "Either call generate, or set dataset_id, not both"
            # (or fix this function so that it can be called twice)"
            self.dataset_id = dataset_id
            self.data_path = os.path.join(self.fm.get_data_path(), self.dataset_id)
        graph = nx.read_gpickle(os.path.join(self.data_path, "world.pkl"))

        with open(os.path.join(self.data_path, "real_orders.pkl"), "rb") as f:
            real_orders = pkl.load(f)

        with open(os.path.join(self.data_path, "idle_driver_locations.pkl"), "rb") as f:
            idle_driver_locations = pkl.load(f)

        with open(os.path.join(self.data_path, "onoff_driver_locations.pkl"), "rb") as f:
            onoff_driver_locations = pkl.load(f)

        with open(os.path.join(self.data_path, "dist.pkl"), "rb") as f:
            dist = pkl.load(f)

        try:
            with open(os.path.join(self.data_path, "random_average_original.pkl"), "rb") as f:
                random_average = np.array(pkl.load(f))
        except IOError:
            random_average = None

        return graph, idle_driver_locations, real_orders, onoff_driver_locations, random_average, dist

    def generate(self):
        '''
        Node ids must be sequential
        '''
        self.dataset_id = self.gen_id()
        self.data_path = os.path.join(self.fm.get_data_path(), self.dataset_id)
        self.fm.clean_path(self.data_path)

        dataset_info = self.generators[self.params['dataset_type']]()
        dataset_info["dataset_id"] = self.dataset_id
        dataset_info["tag"] = self.title
        dataset_info["dataset_path"] = self.data_path
        dataset_info.update(self.params)
        return dataset_info

    def generate_dummy(self, dummy_data_param):
        return {"dummy_data_param_received": dummy_data_param}

    def generate_linear(self, n):
        self.G = nx.path_graph(n)
        for n in self.G.nodes():
            self.G.nodes[n]['coords'] = (n, 0)
        self.assign_weight_by_coords()
        nx.write_gpickle(self.G, os.path.join(self.data_path, "world.pkl"))
        self.generate_dist() # important to generate before orders, so that orders can use dist
        self.generate_orders()
        self.generate_drivers()
        return {}

    def generate_grid(self, n):
        self.G = nx.grid_2d_graph(n, n)
        for n in self.G.nodes():
            self.G.nodes[n]['coords'] = (n[0], n[1])
        self.G = nx.convert_node_labels_to_integers(self.G)
        self.assign_weight_by_coords()

        # if self.params["order_distr"] == "airport":
        #     new_edges = []
        #     for i in range(1,len(self.G)-1):
        #         new_edges.append((0,i))
        #         new_edges.append((len(self.G)-1, i))
        #     self.G.add_edges_from(new_edges)
        #     logging.info("Airport distribution: ")
        #     logging.info("{}".format([self.G.degree(n) for n in self.G.nodes()]))

        nx.write_gpickle(self.G, os.path.join(self.data_path, "world.pkl"))

        self.generate_dist() # important to generate before orders, so that orders can use dist
        self.generate_orders()
        self.generate_drivers()
        return {}

    def generate_hexagon(self, n):
        info = self.generate_grid(n)
        extra_edges = []
        for i in range(n*n):
            x = i % n
            y = i // n
            assert self.G.nodes[i]['coords'] == (y,x), "{}, {}".format(self.G.nodes[i]['coords'], (x,y))
            neighs = [(x-1,y-1),(x-1,y+1),(x+1,y-1),(x+1,y+1)]
            neighs = [x for x in neighs if x[0] >= 0 and x[0] < n and x[1] >= 0 and x[1] < n]
            neigh_ids = [x[1]*n+x[0] for x in neighs]
            extra_edges += [(i, n) for n in neigh_ids]
        self.G.add_edges_from(extra_edges)
        nx.write_gpickle(self.G, os.path.join(self.data_path, "world.pkl"))
        return info

    def assign_weight_by_coords(self):
        for e in self.G.edges():
            self.G[e[0]][e[1]]['weight'] = euclidean(self.G.nodes[e[0]]['coords'], self.G.nodes[e[1]]['coords'])

    def generate_drivers(self):
        idle_driver_locations = np.zeros((self.params["time_periods"], len(self.G)), dtype=int)
        leftover = self.params["number_of_cars"] - np.sum([int(self.params["number_of_cars"]/len(self.G))] * len(self.G))
        for t in np.arange(self.params["time_periods"]):
            idle_driver_locations[t, :] = [int(self.params["number_of_cars"]/len(self.G))] * len(self.G) # uniform
            idle_driver_locations[t, -1] += leftover

        # only city_name = 0 is used in simulation
        with open(os.path.join(self.data_path, "idle_driver_locations.pkl"), "wb") as f:
            pkl.dump(idle_driver_locations, f)

        onoff_driver_locations = np.zeros((self.params["time_periods"], len(self.G), 2))

        with open(os.path.join(self.data_path, "onoff_driver_locations.pkl"), "wb") as f:
            pkl.dump(onoff_driver_locations, f)

    def generate_dist(self):
        dist = np.zeros((len(self.G), len(self.G)))
        for i in range(len(self.G)):
            for j in range(i):
                dist[i,j] = euclidean(self.G.nodes[i]['coords'], self.G.nodes[j]['coords'])
                dist[j,i] = dist[i,j]
        self.dist = dist
        with open(os.path.join(self.data_path, "dist.pkl"), "wb") as f:
            pkl.dump(dist, f)


    def generate_orders(self):
        random_average = []
        hours = self.params["time_periods"] // self.params["time_periods_per_hour"]
        assert(self.params["time_periods"] % self.params["time_periods_per_hour"] == 0)

        for hour in range(hours):
            r = self.get_random_average_orders(self.params["order_distr"], 
                                                self.params["orders_density"], 
                                                self.G, self.params.get('n', None))
            random_average.append(r)
            expected_demand = np.sum(random_average[-1],axis=1) # summation over destination

        real_orders = []
        for tt in np.arange(self.params["days"] * self.params["time_periods"]):
            hour = (tt % self.params["time_periods"]) // self.params["time_periods_per_hour"]
            trips_per_cell = np.random.poisson(random_average[hour])
            for i in range(len(self.G)):
                for j in range(len(self.G)):
                    for trip in range(trips_per_cell[i,j]):
                        # origin grid, destination grid, start time, duration, price
                        pprice = self.dist[i,j]
                        if self.params["order_distr"] == 'airport' and i == 0:
                            pprice *= 10
                        #assert i == 0 or i == len(self.G)-1
                        #assert j != 0 and j != len(self.G)-1
                        real_orders.append([i, j, tt, max([int(pprice),1]), pprice]) # time required for travelling is equal to number of hops, equal to price

        with open(os.path.join(self.data_path, "real_orders.pkl"), "wb") as f:
            pkl.dump(real_orders, f)

        with open(os.path.join(self.data_path, "random_average_original.pkl"), "wb") as f:
            pkl.dump(random_average, f)

    @staticmethod
    def get_random_average_orders(distr_type, density, G, n = None):
        random_average = None
        N = len(G)

        if distr_type == "centered":
            assert(n is not None)
            random_average = np.ones((N,N))
            for i in range(N):
                for j in range(N):
                    p_source = 1 - (abs((n-1)/2 - i % n)/n) - (abs((n-1)/2 - i // n)/n)
                    p_target = 1 - (abs((n-1)/2 - j % n)/n) - (abs((n-1)/2 - j // n)/n)
                    random_average[i,j] *= (p_source*0.8 + p_target*0.2) / 2
            random_average = random_average / np.max(random_average)
            random_average *= density * (1.5 - np.random.random())

        if distr_type == "airport":
            random_average = np.zeros((N,N))
            random_average[0,:] = np.random.random((N,))*density
            random_average[N-1,:] = np.random.random((N,))*density
            random_average[0,0] = 0
            random_average[N-1,N-1] = 0
            random_average[0,N-1] = 0
            random_average[N-1,0] = 0

        if distr_type == "star":
            x_lim = (10000000, -1)
            y_lim = (10000000, -1)
            for n in G.nodes(data=True):
                c = n[1]['coords']
                x_lim = (min(x_lim[0], c[0]), max(x_lim[1], c[0]))
                y_lim = (min(y_lim[0], c[1]), max(y_lim[1], c[1]))

            center_node_coords = ((x_lim[1] - x_lim[0])//2, (y_lim[1] - y_lim[0])//2)
            center = None
            corners = []
            for n in G.nodes(data=True):
                c = n[1]['coords']
                if c == center_node_coords:
                    center = n[0]
                if (c[0] in x_lim and c[1] in y_lim) and (n[0] not in corners) and (n[0] != center):
                    corners.append(n[0])
            assert center is not None, "No node with coords {}".format(center_node_coords)
            assert len(corners) > 0, "No corners for x_lim {} and y_lim {}".format(x_lim, y_lim)

            random_average = np.zeros((N,N))
            for i in corners:
                random_average[i, center] = 1
            random_average *= density

        if distr_type == "uniform":
            random_average = np.random.random((N,N))*density

        if random_average is None:
            raise Exception("Invalid customer distribution type")

        random_average = np.abs(random_average)
        np.fill_diagonal(random_average, 0)
        return random_average

    ### Real-World

    def generate_chicago(self, sparsity):
        '''
        @input:
            -   sparsity creates a graph with subset of nodes, where each <sparsity>-th node is taken
        '''
        g_path = os.path.join(self.fm.get_root_path(), "chicago_g_new.pkl")
        self.G = nx.read_gpickle(g_path)
        # take subsample
        node_samples = np.arange(0, len(self.G), sparsity)
        new_node_index = {}
        for i in range(len(node_samples)):
            new_node_index[node_samples[i]] = i

        self.G = nx.Graph(self.G.subgraph(node_samples))
        self.G = nx.convert_node_labels_to_integers(self.G)
        for i in range(len(self.G)):
            assert(i in self.G)
        nx.write_gpickle(self.G, os.path.join(self.data_path, "world.pkl"))
        # self.generate_dist() -- not needed here anymore
        with open(os.path.join(self.data_path, "dist.pkl"), "wb") as f:
            pkl.dump({}, f) # save empty

        # generate orders -- take first days as a sample
        filtered_orders = []
        with open(os.path.join(self.fm.get_root_path(), "mapped_orders_fixed.csv"), "r") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for r in spamreader:
                if int(r[2]) > self.params["days"]*self.params['time_periods']:
                    break
                source = int(float(r[0]))
                dest = int(float(r[1]))
                if source not in new_node_index or dest not in new_node_index:
                    continue
                filtered_orders.append((new_node_index[source], new_node_index[dest], int(float(r[2])), int(float(r[3])), int(float(r[4]))))
                assert(filtered_orders[-1][3] > 0)
                # <source, destination, time, length, price>

        with open(os.path.join(self.data_path, "real_orders.pkl"), "wb") as f:
            pkl.dump(filtered_orders, f)

        # generate drivers
        with open(os.path.join(self.fm.get_root_path(), "mapped_drivers_fixed.pkl"), "rb") as f:
            drivers = pkl.load(f)

        driver_cells = []
        for i in range(len(drivers)):
            driver_cells += [i]*int(drivers[i])
        driver_cells_sample = np.random.choice(driver_cells, int(self.params['driver_sampling_multiplier']*np.sum(drivers)))
        driver_cells_dict = Counter(driver_cells_sample)
        drivers = np.zeros(drivers.shape)
        for k, val in driver_cells_dict.items():
            drivers[k] = val

        idle_driver_locations = np.zeros((self.params["time_periods"], len(self.G)), dtype=int)
        for t in np.arange(self.params["time_periods"]):
            idle_driver_locations[t, :] = drivers[node_samples]

        with open(os.path.join(self.data_path, "idle_driver_locations.pkl"), "wb") as f:
            pkl.dump(idle_driver_locations, f)

        onoff_driver_locations = np.zeros((self.params["time_periods"], len(self.G), 2))

        with open(os.path.join(self.data_path, "onoff_driver_locations.pkl"), "wb") as f:
            pkl.dump(onoff_driver_locations, f)

        extra_info = {
            "number_of_cars": sum(idle_driver_locations[:,0])
        }
        return extra_info
