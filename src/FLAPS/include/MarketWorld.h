//
// Created by Alvis Logins on 2019-04-17.
//

#ifndef MACAU_MARKETWORLD_H
#define MACAU_MARKETWORLD_H

#include "World.h"
#include "Network.h"

class MarketWorld: public World {
    Network net;
    std::vector<std::vector<unsigned long>> market_to_node_ids;
    std::vector<unsigned long> node_id_to_market;
    std::vector<std::vector<double>> customer_appearance_probability;
    int k;

    void all_states() {
        // calculate probabilities to all possible states

        // enumerate states

        // probability of a state might be small, but not so small given large number of possible outcomes, even from one state

        // maybe there are some dominant states

        // technique of grouping states to reduce their amount, or coarsening the function
        // how to reduce amount by grouping states, how to group states... a state can be grouped as popular and non-popular?
        // but we need to calculate fare based on where and from where it goes... but does it matter? no, fare depends on the difference
        // how far it is.
        // like a radial? hot-not-hot
        // so we have an assumption that taxi do not go to non-popular locations?

        // the q learning approach is when states are grouped, we may group them by appearance probability? state->action. on a grid.
        // state is not cell-id, but what is in the cell. in our case it is the probabilistic appearance of customers.

        // since we can get from any market to any other market it makes sense to group them by appearence, but the reward is different...

        // if we go from one market to any other market then it should be simpler than MDP, it should be a simple multiple matrix multiplication
        // and its also up to preprocessing, not so interesting... the interesting part is how to merge multiple agents together
        // in a bimatching fashion

        // lets say we calculate probabilities of appearing in other sets...
        // p = *[p of all non-zero elements in binvec]
        // they are appearing in time, I need to find who appears first.
        // simple multiplication of matrices will give probability for a policy and action, with no reward?

        // why not simple pagerank?
    }

public:
    MarketWorld(Network& _net, unsigned long markets, std::vector<unsigned long>& _node_id_to_market, int k = 1):
            node_id_to_market(_node_id_to_market), net(_net), k(k) {

        market_to_node_ids.clear();
        market_to_node_ids.resize(markets);
        for (unsigned long i = 0; i < node_id_to_market.size(); i++) {
            market_to_node_ids[node_id_to_market[i]].push_back(i);
        }

        customer_appearance_probability.resize(markets);
        for (long i = 0; i < markets; i++) {
            customer_appearance_probability[i].resize(markets, 0);
        }
    }

    void set_customer_appearance(unsigned long i, unsigned long j, double p) {
        customer_appearance_probability[i][j] = p;
    }

    unsigned long get_state_count() override {
        return market_to_node_ids.size()*((unsigned long)pow(2,k)*(market_to_node_ids.size() - k + 1)); // one for all other zeroes
    }

    unsigned long get_action_count() override {
        return market_to_node_ids.size();
    }

    std::vector<std::pair<unsigned long, double>> get_next_states_with_probs(unsigned long state_id, unsigned long action) override {

    }

    double get_reward(unsigned long state_id, unsigned long action) override {
    }

    bool is_done(unsigned long state) override {
    }

    std::pair<std::vector<bool>, unsigned long> get_target_cells_by_state_id(unsigned long state_id) {
        std::vector<bool> cells(market_to_node_ids.size(), false);
        // first n bits are target availability, where n is number of cells
        std::vector<bool> target_availability(market_to_node_ids.size());
        unsigned long state_id_shifted = state_id;
        for (auto&& bit : target_availability) {
            unsigned long last_bit = state_id_shifted & 1;
            state_id_shifted = state_id_shifted >> 1;
            bit = last_bit == 1;
        }
        // the rest of state_id_shifted is id of current cell
        unsigned long current_cell_id = state_id_shifted;
        return std::make_pair(target_availability, current_cell_id);
    }
};

#endif //MACAU_MARKETWORLD_H
