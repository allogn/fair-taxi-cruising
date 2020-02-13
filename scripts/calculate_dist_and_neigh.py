import sys, os
sys.path.append(os.path.dirname(os.path.abspath('')))
sys.path.append(os.path.join(os.path.abspath(''),'..','scripts'))
sys.path.append(os.path.join(os.path.abspath(''),'..','src'))

import pandas as pd
import pickle as pkl
import networkx as nx
import numpy as np
import math
from tqdm import tqdm

DATA_PATH = '/Users/au624601/PhD/data/taxi/'

G = nx.read_graphml(os.path.join(DATA_PATH, "chicago_processed.graphml"))
G = G.to_undirected() # Maps may not be accurate, so let find undirected paths only

census_centroids = pkl.load(open(os.path.join(DATA_PATH, "census_centroids.pkl"), "rb"))
distances = {}
next_census = {}
for k1 in tqdm(census_centroids):  #indexed by name10
    sp = nx.single_source_dijkstra_path_length(G, census_centroids[k1], weight="length")
    for k2 in census_centroids:
        if k1 == k2 or (k1,k2) in distances:
            continue
        if census_centroids[k2] not in sp:
            p_length = -1
        else:
            p_length = sp[census_centroids[k2]]
        distances[(k1, k2)] = p_length
pkl.dump(distances, open(os.path.join(DATA_PATH, "distances_new.pkl"), "wb"))
