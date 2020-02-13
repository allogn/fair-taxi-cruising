//
// Created by Alvis Logins on 2019-04-19.
//

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GraphTest
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <vector>

#include "Generator.h"
#include "GridParams.h"

namespace fs = boost::filesystem;
namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(InitGraph) {
    GridParams params;
    params.sqrt_n = 4;
    Generator gen = Generator();
    Network g = gen.generate(GRID, &params);
    BOOST_TEST(g.get_size() == 16);
    BOOST_TEST(g.get_edge_count() == 48);
}