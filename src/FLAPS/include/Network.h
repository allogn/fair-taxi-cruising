//
// Created by Alvis Logins on 2019-04-17.
//

#ifndef MACAU_NETWORK_H
#define MACAU_NETWORK_H

#include <vector>
#include "NetworkParams.h"

class Network {
    std::vector<unsigned long> out_neighbor_list;
    std::vector<unsigned long> out_degree_index;
    std::vector<int> out_weights;
    NetworkParams* p;
public:
    Network(NetworkParams* _p): p(_p) {};
    void set_edgelist(std::vector<unsigned long> edgelist, std::vector<int> weights) {
        assert(edgelist.size() % 2 == 0);
        assert(edgelist.size() / 2 == p->m);

        // count out degree
        std::vector<unsigned int> outdegree(p->n, 0);
        for (unsigned long i = 0; i < edgelist.size(); i+=2) {
            outdegree[edgelist[i]]+=1;
        }
        out_degree_index.resize(p->n);
        unsigned long sum = 0;
        for (unsigned long i = 0; i < p->n; i++) {
            sum += outdegree[i];
            out_degree_index[i] = sum;
        }
        assert(sum == p->m);
        assert(weights.size() == p->m);
        out_neighbor_list.resize(p->m);
        out_weights.resize(p->m);
        for (unsigned long i = 0; i < edgelist.size(); i+=2) {
            unsigned long edge_index = out_degree_index[edgelist[i]]-outdegree[edgelist[i]];
            out_neighbor_list[edge_index] = edgelist[i+1];
            out_weights[edge_index] = weights[i/2];
            edgelist[i]--;
        }
    }

    unsigned long get_size() {
        return p->n;
    }

    unsigned long get_edge_count() {
        return p->m;
    }
};

#endif //MACAU_NETWORK_H
