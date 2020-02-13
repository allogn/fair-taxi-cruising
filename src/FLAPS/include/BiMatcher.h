//
// Created by Alvis Logins on 2019-03-21.
//

#ifndef MACAU_BIMATCHER_H
#define MACAU_BIMATCHER_H

#include "Graph.h"
#include "Oracle.h"
#include "SubMin.h"

class BiMatcher {
    Graph* g;
    Oracle* eo;
    SubMin* submin_solver;
public:
    BiMatcher(Graph* g) {
        this->g = g;
        eo = new Oracle(g);
        submin_solver = new SubMin(eo);
    }
    ~BiMatcher() {
        delete submin_solver;
        delete eo;
    }

    void run() {

    }
};

#endif //MACAU_BIMATCHER_H
