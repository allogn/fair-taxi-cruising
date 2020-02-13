'''
This takes pbf file of a region, downloaded from OSM (https://download.geofabrik.de/), takes city name, 
and extracts cleaned (largest weaked component of road ways) networkx graph
Then the graph is exported to a graphml file
'''

import osmium
import networkx as nx
from geopy.geocoders import Nominatim
import sys
import re

pruned_ways = {"unrelated",'cycleway','pedestrian','footway','steps','path','railway',"building"}

class CounterHandler(osmium.SimpleHandler):
    def __init__(self):
        osmium.SimpleHandler.__init__(self)
        self.num_nodes = 0

    def node(self, n):
        self.num_nodes += 1


class MyCounterHandler(osmium.SimpleHandler):
    def __init__(self, node_dict, edgelist, city_bbox):
        self.edgelist = edgelist
        self.node_dict = node_dict
        self.city_bbox = city_bbox
        self.total_nodes = 0
        self.total_edges = 0
        osmium.SimpleHandler.__init__(self)

    def node(self, n):
        self.total_nodes += 1

    def is_point_inside_bbox(self, lon, lat):
        return lon >= self.city_bbox[2] and lon <= self.city_bbox[3] and lat >= self.city_bbox[0] and lat <= self.city_bbox[1]

    def way(self, w):
        self.total_edges += len(w.nodes)
        for t in w.tags:
            if (t.k in pruned_ways) or (t.v in pruned_ways):
                return
        for i in range(len(w.nodes)-1):
            if not self.is_point_inside_bbox(w.nodes[i].lon, w.nodes[i].lat):
                continue
            
            if (w.nodes[i].ref not in self.node_dict):
                self.node_dict[w.nodes[i].ref] = (w.nodes[i].lon, w.nodes[i].lat)
            self.edgelist.append((w.nodes[i].ref, w.nodes[i+1].ref))
        
        last_point = w.nodes[len(w.nodes)-1]
        if (not self.is_point_inside_bbox(last_point.lon, last_point.lat)) and ((last_point.ref not in self.node_dict)):
            self.node_dict[last_point.ref] = (last_point.lon, last_point.lat)
            
    def relation(self, r):
        pass
    
def query_city_bbox(cityname):
    # geocode city
    geolocator = Nominatim(user_agent="my-application")
    location = geolocator.geocode(cityname)
    if location == None:
        raise Exception("City not found")
    bbox_arr = location.raw['boundingbox']
    bbox = (float(bbox_arr[0]), float(bbox_arr[1]), float(bbox_arr[2]), float(bbox_arr[3]))
    return bbox
        
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: getCleanedGraph.py <osm file> <city name> <out nx graph file>")
        exit()
        
    city_bbox = query_city_bbox(sys.argv[2])
    
    nodes = {}
    edges = []
    h = MyCounterHandler(nodes, edges, city_bbox)

    h.apply_file(sys.argv[1], locations=True)
    print("Unpruned nodes and edges: {} / {}".format(h.total_nodes, h.total_edges))
    G = nx.DiGraph()
    G.add_nodes_from([(k, {"lon":nodes[k][0], "lat":nodes[k][1]}) for k in nodes])
    G.add_edges_from(edges)
    gg = [G.subgraph(c) for c in nx.weakly_connected_components(G)]
    G = max(gg, key=lambda x: len(x))
    G = G.subgraph([n[0] for n in G.nodes(data=True) if len(n[1]) > 0]) # remove nodes without location
    nx.write_graphml(G, sys.argv[3])
    print("Final: {} / {}".format(G.number_of_nodes(), G.number_of_edges()))
    