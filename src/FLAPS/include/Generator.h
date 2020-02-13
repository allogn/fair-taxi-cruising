//
// Created by Alvis Logins on 2019-04-19.
//

#ifndef MACAU_GENERATOR_H
#define MACAU_GENERATOR_H

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <exception>

#include "Network.h"
#include "GridParams.h"

typedef enum {
    GRID
} GraphType;

class Generator {
public:
    explicit Generator(unsigned int seed = time(nullptr)) {
        srand (seed);
    };

    Network generate(GraphType gt, NetworkParams* p) {
        if (gt == GRID) {
            return get_grid(p);
        }
        throw std::runtime_error("Graph type is not implemented");
    }

    Network get_grid(NetworkParams* p_base) {
        //rand()%100
        auto p = dynamic_cast<GridParams*>(p_base);
        Network net = Network(p);
        std::vector<unsigned long> edgelist;
        std::vector<int> weights;
        p->n = p->sqrt_n*p->sqrt_n;
        p->m = p->sqrt_n*(p->sqrt_n-1)*4;
        for (unsigned long i = 0; i < p->n; i++) {
            if (i % p->sqrt_n != 0) {
                edgelist.push_back(i);
                edgelist.push_back(i-1);
                weights.push_back(1);
            }
            if (i % p->sqrt_n != p->sqrt_n-1) {
                edgelist.push_back(i);
                edgelist.push_back(i+1);
                weights.push_back(1);
            }
            if (i > p->sqrt_n-1) {
                edgelist.push_back(i);
                edgelist.push_back(i-p->sqrt_n);
                weights.push_back(1);
            }
            if (i < p->n-p->sqrt_n) {
                edgelist.push_back(i);
                edgelist.push_back(i+p->sqrt_n);
                weights.push_back(1);
            }
        }
        net.set_edgelist(edgelist, weights);
        return net;
    }
};

#endif //MACAU_GENERATOR_H
