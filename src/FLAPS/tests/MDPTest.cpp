//
// Created by Alvis Logins on 2019-04-17.
//

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE MDPTest
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <vector>

#include "MDP.h"
#include "GridWorld.h"

namespace fs = boost::filesystem;
namespace utf = boost::unit_test;

// Have to make it a macro so that it reports exact line numbers when checks fail.
#define CHECK_CLOSE_COLLECTION(aa, bb, tolerance) { \
    using std::distance; \
    using std::begin; \
    using std::end; \
    auto a = begin(aa), ae = end(aa); \
    auto b = begin(bb); \
    BOOST_REQUIRE_EQUAL(distance(a, ae), distance(b, end(bb))); \
    for(; a != ae; ++a, ++b) { \
        BOOST_CHECK_CLOSE(*a, *b, tolerance); \
    } \
}

BOOST_AUTO_TEST_CASE(InitMDP, *utf::tolerance(0.1)) {
    auto world = GridWorld();
    auto solver = MDP(world);
    std::vector<std::vector<double>> uniform_policy(world.get_state_count());
    for (int i = 0; i < world.get_state_count(); i++) {
        uniform_policy[i] = std::vector<double>(world.get_action_count());
        for (int j = 0; j < world.get_action_count(); j++) {
            uniform_policy[i][j] = 1./world.get_action_count();
        }
    }
    auto p = solver.policy_eval(uniform_policy);
    std::vector<double> p_correct({0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0});
    CHECK_CLOSE_COLLECTION(p, p_correct, 0.01);
}