//
// Created by Alvis Logins on 2019-03-20.
//

#ifndef MACAU_GRAPH_H
#define MACAU_GRAPH_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <list>

#define NODE_TYPES 2

class Graph {
    std::unordered_map<long, long> node_label_to_index[NODE_TYPES];
    std::vector<long> index_to_node_label[NODE_TYPES];
    std::unordered_set<long> node_set[NODE_TYPES];

    std::vector<double> edge_probabilities;
    std::vector<double> weights;

    long number_of_nodes[NODE_TYPES];

public:
    std::vector<std::list<std::pair<int, long>>> neighbors[NODE_TYPES]; // used in oracle

    Graph() {
        number_of_nodes[0] = 0;
        number_of_nodes[1] = 0;
    }

    void add_node(long node_label, int type) {
        if (node_set[type].find(node_label) == node_set[type].end()) {
            node_set[type].insert(node_label);
            node_label_to_index[type][node_label] = number_of_nodes[type];
            index_to_node_label[type].push_back(node_label);
            number_of_nodes[type]++;
            neighbors[type].emplace_back();
        }
    }

    void add_edge(long a_label, long b_label, double w, double probability) {
        add_node(a_label, 0);
        add_node(b_label, 1);
        assert(probability >= 0 && probability <= 1);
        long edge_id = edge_probabilities.size();
        edge_probabilities.push_back(probability);
        weights.push_back(w);
        long a_ind = node_label_to_index[0][a_label];
        long b_ind = node_label_to_index[1][b_label];
        neighbors[0][a_ind].push_back(std::make_pair(b_ind, edge_id));
        neighbors[1][b_ind].push_back(std::make_pair(a_ind, edge_id));
    }

    inline long get_size() {
        return number_of_nodes[0] + number_of_nodes[1];
    }

    inline unsigned long get_edge_count() {
        return weights.size();
    }

    inline double get_weight(double edge_id) {
        return weights[edge_id];
    }

    inline double get_probability(double edge_id) {
        return edge_probabilities[edge_id];
    }

    inline unsigned long get_neighbor_count(long node_label, int type) {
        return neighbors[type][node_label_to_index[type][node_label]].size();
    }

    inline long get_id_by_label(long label, int type) {
        return node_label_to_index[type][label];
    }
};

#endif //MACAU_GRAPH_H
