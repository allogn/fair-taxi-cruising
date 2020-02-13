import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
import math as math
import subprocess
import sys
import os
import json
import pickle as pkl
DATA_PATH = os.path.join(os.environ['ALLDATA_PATH'],'taxi')

def transform_point(point, lat0, lon0):
    lat = point[1]
    lon = point[0]
    dlat = lat - lat0;
    dlon = lon - lon0;
    latitudeCircumference = 40075160. * math.cos(lat0 * math.pi/180.0);
    resX = dlon * latitudeCircumference / 360. ;
    resY = dlat * 40008000. / 360. ;
    return [resX, resY]

def transform_coords(polygon):
    '''
    Transforms all points in shapely polygons
    '''
    new_polygon = []
    for point in polygon:
        new_polygon.append(transform_point(point))
    return new_polygon
