//
// Created by Alvis Logins on 2019-03-14.
//

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GraphTest
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <vector>

#include "Graph.h"

namespace fs = boost::filesystem;
namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(InitGraph) {
    Graph g;
    g.add_node(1,0);
    g.add_edge(1,2,4,0.4);
    g.add_edge(0,0,2,0.1);
    g.add_node(0,1);
    g.add_edge(2,1,0,0.2);
    BOOST_TEST(g.get_probability(2) == 0.2);
    BOOST_TEST(g.get_weight(0) == 4);
    g.add_edge(0,2,1,1);
    BOOST_TEST(g.get_neighbor_count(0,0) == 2);
    BOOST_TEST(g.get_edge_count() == 4);
    BOOST_TEST(g.get_size() == 6); // 0: 1,1,0,2,0 -> 3 ; 1: 2,0,0,1,2 -> 3

    BOOST_TEST(g.get_id_by_label(0,0) == 1);
    for (auto& n : g.neighbors[0][1]) // node_id 1 should correspond to label 0, by the order of insertion
    {
        BOOST_TEST(n.first == g.get_id_by_label(0,1));
        BOOST_TEST(n.second == 1);
        break;
    }
}