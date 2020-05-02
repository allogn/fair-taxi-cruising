import numpy as np
import networkx as nx

from framework.solvers.SplitSolver import SplitSolver
from framework.Generator import Generator

class TestSplitSolver:

    def test_build_worlds(self):
        n = 6
        G = nx.grid_2d_graph(n, n, create_using=nx.DiGraph())
        for n in G.nodes():
            G.nodes[n]['coords'] = (n[0], n[1])
        G = nx.convert_node_labels_to_integers(G)
        superworld = SplitSolver.build_superworld(G, 4)
        assert len(superworld) == 4
        for n in superworld.nodes(data=True):
            assert abs(len(n[1]['nodes']) - 9) <= 3 # might not be equal
            for n2 in n[1]['nodes']:
                assert G.has_node(n2)
            assert nx.is_weakly_connected(G.subgraph(n[1]['nodes']))
        assert superworld.number_of_edges() >= 4
        # might be greater than 8, because splitting might not be exactly square-based (due to hilbert approximation)
        for n in G.nodes(data=True):
            assert 'supernode' in n[1]