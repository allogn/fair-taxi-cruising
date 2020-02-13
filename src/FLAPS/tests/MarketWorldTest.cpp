//
// Created by Alvis Logins on 2019-04-23.
//

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE MarketWorldTest
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <vector>

#include "MarketWorld.h"
#include "Generator.h"

namespace fs = boost::filesystem;
namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(StateIdToBinVecTest) {
    auto gen = Generator();
    auto gridparams = GridParams();
    gridparams.sqrt_n = 2;
    auto net = gen.generate(GRID, &gridparams);
    std::vector<unsigned long> node_id_to_market({0,0,0,1,1,1,2,2,2,2,2,2});
    auto mw = MarketWorld(net, 3, node_id_to_market, 2);
    BOOST_TEST(mw.get_action_count() == 3);
    BOOST_TEST(mw.get_state_count() == 3*(4*2));

    auto binvec_s_pair = mw.get_target_cells_by_state_id(12);
    BOOST_TEST(binvec_s_pair.first == std::vector<bool>({1,0,0}));
    BOOST_TEST(binvec_s_pair.second == 1);
    std::vector<long> state_counts(mw.get_state_count(),0);
    for (int i = 0; i < mw.get_state_count(); i++) {
        binvec_s_pair = mw.get_target_cells_by_state_id(i);
        state_counts[binvec_s_pair.second]++;
        BOOST_TEST(binvec_s_pair.first.size() == state_counts.size());
        for (int j = 0; j < binvec_s_pair.first.size(); j++) {
            if (binvec_s_pair.first[j]) {
                state_counts[j]++;
            }
        }
    }
    for (int i = 0; i < mw.get_state_count(); i++) {
        BOOST_TEST(state_counts[i] == )
    }
}