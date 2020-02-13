//
// Created by Alvis Logins on 2019-03-14.
//

#ifndef MACAU_SUBMIN_H
#define MACAU_SUBMIN_H

#include <limits>
#include <vector>
#include <list>
#include <exception>
#include <math.h>
#include <random>
#include <algorithm>

#include "BST.h"
#include "Oracle.h"
#include "helpers.h"

typedef enum {BRUTEFORCE, LOVASZ} SolverType;
typedef std::list<std::pair<long, double>> SparseVec;

class SubMin {
    SolverType solver_type;
    std::list<long> S;
    unsigned long problem_size;
    Oracle* EO;
    double objective;

    void bruteforce() {
        double best_obj = MAXFLOAT;
        int counter = 0;
        while (counter < pow(2, problem_size)) {
            std::list<long> selected_indexes;
            int i = 0;
            int counter2 = counter;
            while (counter2 > 0) {
                if ((counter2 & 1) == 1) {
                    selected_indexes.push_back(i);
                }
                i++;
                counter2 = counter2 >> 1;
            }
            double obj = EO->eval(selected_indexes.begin(), selected_indexes.end());
            if (obj < best_obj) {
                best_obj = obj;
                S = selected_indexes;
            }
            counter++;
        }
        assert(best_obj < MAXFLOAT);
        objective = best_obj;
    }

    /* The BST solution */
    double epsilon;
    double N;
    double T;
    double nu;
    BST bst;

    std::vector<double> get_lovasz_subgrad() {
        std::list<long> sorted_P = bst.get_desc_sorted_indexes();
        std::vector<double> f_val = EO->eval_seq(sorted_P.begin(), sorted_P.end());
        std::vector<double> g(f_val.size());
        g[0] = f_val[0];
        for (long i = 0; i < f_val.size(); i++) {
            g[i] = f_val[i] - f_val[i-1];
        }
        return g;
    }

    void transform_lovasz_to_s() {
        // sort x
        // eval all f
        // get smallest l
    }

    long get_random_index(std::vector<double>& prob_distr) {
        std::default_random_engine generator;
        std::discrete_distribution<int> distribution(prob_distr.begin(), prob_distr.end());
        std::vector<double> vec(1);
        std::vector<int> indices(vec.size());
        std::generate(indices.begin(), indices.end(), [&generator, &distribution]() { return distribution(generator); });
        //std::transform(indices.begin(), indices.end(), vec.begin(), [&samples](int index) { return samples[index]; });
        return indices[0];
    }

    void init_lovasz() {
        N = 10*problem_size*pow(log(problem_size),2)/(epsilon*epsilon);
        T = pow(problem_size, 1./3.);
        std::vector<double> x(problem_size, 0); //todo move to bst
        bst.insert(x);
        nu = sqrt(problem_size)/18; // M = 1
    }

    void run_lovasz() {
        for (long batch_iter = 0; batch_iter < (long)N/T; batch_iter++) {
            run_batch();
        }
        transform_lovasz_to_s();
    }

    void run_batch() {
        std::vector<double> lovasz_subgrad = get_lovasz_subgrad();
        double lovasz_norm = 0;
        for (const auto& v : lovasz_subgrad) lovasz_norm += abs(v);

        long sampled_index = get_random_index(lovasz_subgrad);
        double z_val = lovasz_norm * sgn(lovasz_subgrad[sampled_index]);
        SparseVec z;
        z.emplace_back(sampled_index, z_val);

        for (long iteration = 0; iteration < T; iteration++) {
            run_batch_iteration(z, iteration);
        }
    }

    void run_batch_iteration(SparseVec& z, long l) {
        SparseVec e_pos;
        SparseVec e_neg;
        double x[1];
        for (auto& z_val : z) {
            long ind = z_val.first;
            double val;
            if (z_val.second > 0) { //todo check if sgn(z_val) == sgn(g)
                val = std::min(x[ind], nu*z_val.second);
            } else {
                val = std::max(x[ind]-1, nu*z_val.second);
            }
            if (val > 0) {
                e_pos.emplace_back(ind, val);
            }
            if (val < 0) {
                e_neg.emplace_back(ind, val);
            }
        }
//        Sample(z, e_pos, l);
//        Sample(z, e_neg, l);
    }

    void Sample(SparseVec& z, SparseVec& e, long l) {

    }

public:
    SubMin(Oracle* EO, SolverType mode=BRUTEFORCE) {
        this->EO = EO;
        problem_size = EO->get_problem_size();
        solver_type = mode;
    }

    void run() {
        switch (solver_type) {
            case BRUTEFORCE:
                bruteforce();
                break;
            case LOVASZ:
                init_lovasz();
                break;
            default:
                throw std::runtime_error("Solver Type is not implemented");
        }
    }

    inline double get_objective() {
        return objective;
    }

    inline std::list<long> get_optimal_normalized_set() {
        return S;
    }
};

#endif //MACAU_SUBMIN_H
