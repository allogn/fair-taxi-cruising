//
// Created by Alvis Logins on 2019-04-17.
//

#ifndef MACAU_MDP_H
#define MACAU_MDP_H

#include <vector>

#include "World.h"

class MDP {
    World& world;
    double discount_factor;
    double epsilon;
public:
    MDP(World& w, double discount_factor = 1.0, double epsilon = 0.00001):
        world(w),
        discount_factor(discount_factor),
        epsilon(epsilon) {

    }

    ~MDP() = default;

    std::vector<double> policy_eval(std::vector<std::vector<double>>& policy) {
        std::vector<double> states(world.get_state_count(), 0);
        std::vector<double> states_new(world.get_state_count(), 0);
        while (true) {
            double delta = 0;
            for (unsigned long s = 0; s < world.get_state_count(); s++) {
                if (world.is_done(s)) continue;

                std::vector<double>& action_probs = policy[s];
                double v_fn = 0; // bellman expectation

                for (int a = 0; a < world.get_action_count(); a++) {

                    std::vector<std::pair<unsigned long, double>> next_states_with_probs = world.get_next_states_with_probs(s, a);
                    double reward = world.get_reward(s, a);

                    double expected_next_states = 0;
                    for (const auto& next_state_with_prob : next_states_with_probs) {
                        expected_next_states += next_state_with_prob.second * states[next_state_with_prob.first];
                    }
                    v_fn += action_probs[a] * (reward + discount_factor * expected_next_states);
                }

                double new_delta = abs(v_fn - states[s]);
                delta = (delta > new_delta) ? delta : new_delta;
                states_new[s] = v_fn;

            }
            states = states_new;
            if(delta < epsilon) break;
        }

        return states;
    }
};

#endif //MACAU_MDP_H
