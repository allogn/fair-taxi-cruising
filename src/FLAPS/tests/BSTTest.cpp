//
// Created by Alvis Logins on 2019-03-14.
//

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE BSTTest
#include <boost/test/unit_test.hpp>

#include "BST.h"

//BOOST_AUTO_TEST_CASE(InitBST) {
//    BST bst;
//    bst.check_tree_structure();
//    std::vector<Node*> nodes;
//    nodes.push_back(bst.insert(5));
//    bst.check_tree_structure();
//    nodes.push_back(bst.insert(542345));
//    bst.check_tree_structure();
//    nodes.push_back(bst.insert(-1));
//    bst.check_tree_structure();
//    nodes.push_back(bst.insert(0.));
//    bst.check_tree_structure();
//    nodes.push_back(bst.insert(0.));
//    bst.check_tree_structure();
//    nodes.push_back(bst.insert(5));
//    bst.check_tree_structure();
//    BOOST_REQUIRE_EQUAL(bst.get_size(), 6);
//
//    for (auto i : nodes) {
//        bst.remove(i);
//        bst.check_tree_structure();
//    }
//    BOOST_REQUIRE_EQUAL(bst.get_size(), 0);
//}

BOOST_AUTO_TEST_CASE(TestVectorInit) {
    BST bst;
    std::vector<double> x({0.1,0.4,0,0.,0.1,1,1,0.4,0.44432});
    bst.insert(x);
    bst.check_tree_structure();
    BOOST_REQUIRE_EQUAL(bst.get_size(), x.size());
    std::list<long> sorted_index = bst.get_desc_sorted_indexes();
    std::list<long> correct({6,5,8,7,1,4,0,3,2});
    BOOST_TEST(sorted_index == correct);
}