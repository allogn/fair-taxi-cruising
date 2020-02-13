//
// Created by Alvis Logins on 2019-03-14.
//

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SubMinTest
#include <boost/test/unit_test.hpp>

#include "SubMin.h"

namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(testBruteforce, *utf::tolerance(0.00001)) {
    Graph g;
    g.add_edge(0,1,1,0.5);
    g.add_edge(0,2,1,0.3);
    g.add_edge(0,3,1,0.2);

    Oracle eo(&g);
    eo.set_active_node(0,0);
    eo.update_current_xe(0,0.01);
    eo.update_current_xe(1,0.3);
    eo.update_current_xe(2,0.2);

    auto solver = SubMin(&eo, BRUTEFORCE);
    solver.run();
    BOOST_TEST(solver.get_objective() == -0.06);
    BOOST_TEST(solver.get_optimal_normalized_set() == std::list<long>({1,2}));

}