//
// Created by Alvis Logins on 2019-03-21.
//

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE OracleTest
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <vector>
#include <list>
#include <iterator>

#include "Oracle.h"

BOOST_AUTO_TEST_CASE(InitOracle) {
    Graph g;
    g.add_edge(1,3,1,0.5);
    g.add_edge(1,4,2,0.4);
    g.add_edge(2,4,10,1);

    auto eo = Oracle(&g);
    eo.set_active_node(g.get_id_by_label(1,0),0);
    eo.update_current_xe(0, 0.1);
    eo.update_current_xe(1, 0.2);

    BOOST_TEST(eo.get_problem_size() == 2);
    std::list<long> edges;
    for (long i = 0; i < eo.get_problem_size(); i++) {
        edges.push_back(i);
    }

    double f_val = eo.eval(edges.begin(), edges.end());
    BOOST_TEST(f_val == (1-(1-0.5)*(1-0.4)) - (0.1 + 0.2));


    auto f_val_seq = eo.eval_seq(edges.begin(), edges.end());
    for (long i = 0; i < f_val_seq.size(); i++) {
        auto end = edges.begin();
        std::advance(end,i+1);
        BOOST_TEST(f_val_seq[i] == eo.eval(edges.begin(), end));
    }
}