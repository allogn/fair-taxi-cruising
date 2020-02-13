//
// Created by Alvis Logins on 2019-03-20.
//

#ifndef MACAU_ORACLE_H
#define MACAU_ORACLE_H

#include <list>

#include "Graph.h"

class Oracle {
    Graph* g;
    std::vector<double> xe;
    std::pair<long,int> active_node;
    std::vector<long> edge_mapping;

public:
    Oracle(Graph* g) {
        this->g = g;
        xe.clear();
        xe.resize(g->get_edge_count(), 0);
    }

    ~Oracle() {}

    void set_active_node(long node, int node_type) {
        edge_mapping.clear();
        active_node = std::make_pair(node, node_type);
        for (const auto& v : g->neighbors[node_type][node]) {
            edge_mapping.push_back(v.second);
        }
    }

    double eval(std::list<long>::iterator P1, std::list<long>::iterator P2) {
        // input - normalized edge list iterator to the beginning and the end
        double p = 1;
        double x_sum = 0;
        while (P1 != P2) {
            p *= 1 - g->get_probability(edge_mapping[(*P1)]);
            x_sum += xe[edge_mapping[(*P1)]];
            P1++;
        }
        return (1 - p) - x_sum;
    }

    std::vector<double> eval_seq(std::list<long>::iterator P1, std::list<long>::iterator P2) {
        std::vector<double> res;
        double p = 1;
        double x_sum = 0;
        while (P1 != P2) {
            p *= 1 - g->get_probability(edge_mapping[(*P1)]);
            x_sum += xe[edge_mapping[(*P1)]];
            P1++;
            res.push_back((1 - p) - x_sum);
        }
        return res;
    }

    unsigned long get_problem_size() {
        return edge_mapping.size();
    }

    inline void update_current_xe(long normalized_xe_index, double value) {
        xe[edge_mapping[normalized_xe_index]] = value;
    }
};

#endif //MACAU_ORACLE_H
