//
// Created by Alvis Logins on 2019-04-17.
//

#ifndef MACAU_GRIDWORLD_H
#define MACAU_GRIDWORLD_H

#include "World.h"

class GridWorld: public World {
public:
    GridWorld() = default;

    unsigned long get_state_count() override {
        return 16;
    }

    unsigned long get_action_count() override {
        return 4;
    }

    std::vector<std::pair<unsigned long, double>> get_next_states_with_probs(unsigned long state, unsigned long action) override {
        unsigned long next_state_id = 0;
        switch (action) {
            case 0:
                if (state % 4 == 0) {
                    next_state_id = state;
                } else {
                    next_state_id = state-1;
                }
                break;
            case 1:
                if (state < 4) {
                    next_state_id = state;
                } else {
                    next_state_id = state-4;
                }
                break;
            case 2:
                if (state % 4 == 3) {
                    next_state_id = state;
                } else {
                    next_state_id = state+1;
                }
                break;
            case 3:
                if (state > 11) {
                    next_state_id = state;
                } else {
                    next_state_id = state+4;
                }
                break;
            default:
                assert(false);
        }
        std::pair<unsigned long, double> p = std::make_pair(next_state_id, 1);
        return std::vector<std::pair<unsigned long, double>>({p});
    }
    double get_reward(unsigned long state, unsigned long action) override {
        return -1;
    }
    bool is_done(unsigned long state) override {
        return (state == 0 || state == 15);
    }
};

#endif //MACAU_GRIDWORLD_H
