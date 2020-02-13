//
// Created by Alvis Logins on 2019-04-17.
//

#ifndef MACAU_WORLD_H
#define MACAU_WORLD_H


typedef struct {
    bool is_state; // action otherwise
    unsigned long id;
} WorldNode;


typedef struct {
    unsigned long source_id;
    unsigned long dest_id;
    double probability;
} WorldEdge;

class World {
    std::vector<WorldNode> nodes;
    std::vector<WorldEdge> edges;
    unsigned long root;
public:
    World() = default;
    virtual unsigned long get_state_count() { return 0; }
    virtual unsigned long get_action_count() { return 0; }
    virtual std::vector<std::pair<unsigned long, double>> get_next_states_with_probs(unsigned long state_id, unsigned long action) {
        return std::vector<std::pair<unsigned long, double>>(); 
    }
    virtual double get_reward(unsigned long state_id, unsigned long action) { return 0; }
    virtual bool is_done(unsigned long state_id) { return true; }
};

#endif //MACAU_WORLD_H
