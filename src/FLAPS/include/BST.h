//
// Created by Alvis Logins on 2019-03-14.
//
// RB Tree implementation derived from https://en.wikipedia.org/wiki/Red%E2%80%93black_tree
//

#ifndef MACAU_BST_H
#define MACAU_BST_H

#include <vector>
#include <map>
#include <list>

typedef enum {RED, BLACK} Color;

typedef struct Node {
    Node* parent = nullptr;
    Node* left = nullptr;
    Node* right = nullptr;
    bool leaf = true;
    Color color = BLACK;
    double key;
    int index;
    Node* left_sorted = nullptr;
    Node* right_sorted = nullptr;
} Node;

class BST {
    // node updates counters
    long created = 0;
    long deleted = 0;

    Node* root;
    int size;
    Node* max_node;
    Node* min_node;

    Node* parent(Node* n) {
        return n->parent; // nullptr for root node
    }

    Node* grandparent(Node* n) {
        Node* p = parent(n);
        if (p == nullptr)
            return nullptr; // No parent means no grandparent
        return parent(p); // nullptr if parent is root
    }

    Node* sibling(Node* n) {
        Node* p = parent(n);
        if (p == nullptr)
            return nullptr; // No parent means no sibling
        if (n == p->left)
            return p->right;
        else
            return p->left;
    }

    Node* uncle(Node* n) {
        Node* p = parent(n);
        Node* g = grandparent(n);
        if (g == nullptr)
            return nullptr; // No grandparent means no uncle
        return sibling(p);
    }

    void rotate_left(Node* n) {
        Node* nnew = n->right;
        Node* p = parent(n);
        assert(!nnew->leaf); // since the leaves of a red-black tree are empty, they cannot become internal nodes
        n->right = nnew->left;
        nnew->left = n;
        n->parent = nnew;
        // handle other child/parent pointers
        if (n->right != nullptr)
            n->right->parent = n;
        if (p != nullptr) // initially n could be the root
        {
            if (n == p->left)
                p->left = nnew;
            else if (n == p->right) // if (...) is excessive
                p->right = nnew;
        }
        nnew->parent = p;
    }

    void rotate_right(Node* n) {
        Node* nnew = n->left;
        Node* p = parent(n);
        assert(!nnew->leaf); // since the leaves of a red-black tree are empty, they cannot become internal nodes
        n->left = nnew->right;
        nnew->right = n;
        n->parent = nnew;
        // handle other child/parent pointers
        if (n->left != nullptr)
            n->left->parent = n;
        if (p != nullptr) // initially n could be the root
        {
            if (n == p->left)
                p->left = nnew;
            else if (n == p->right) // if (...) is excessive
                p->right = nnew;
        }
        nnew->parent = p;
    }

    void insert(Node* n) {
        // insert new node into the current tree
        insert_recurse(root, n);

        // repair the tree in case any of the red-black properties have been violated
        insert_repair_tree(n);

        // find the new root to return
        root = n;
        while (parent(root) != nullptr)
            root = parent(root);
    }

    inline bool cmp(Node* a, Node* b) {
        // return true if a < b
        return (a->key < b->key) || ((a->key == b->key) && (a->index < b->index));
    }

    void insert_recurse(Node* root, Node* n) {
        // recursively descend the tree until a leaf is found
        assert(root != nullptr);
        Node* leaf = root;
        if (!root->leaf && cmp(n, root)) {
            if (!root->left->leaf) {
                insert_recurse(root->left, n);
                return;
            }
            else
                leaf = root->left;
                root->left = n;
        } else if (!root->leaf) {
            if (!root->right->leaf){
                insert_recurse(root->right, n);
                return;
            }
            else
                leaf = root->right;
                root->right = n;
        }

        // insert new node n
        if (root->leaf) {
            n->parent = nullptr;
        } else {
            n->parent = root;
        }

        n->left = leaf;
        n->left->parent = n;

        n->right = new Node();
        created++;
        n->right->parent = n;

        n->color = RED;
    }

    void insert_repair_tree(Node* n) {
        if (parent(n) == nullptr) {
            insert_case1(n);
        } else if (parent(n)->color == BLACK) {
            insert_case2(n);
        } else if (uncle(n) != nullptr && uncle(n)->color == RED) {
            insert_case3(n);
        } else {
            insert_case4(n);
        }
    }

    void insert_case1(Node* n)
    {
        if (parent(n) == nullptr)
            n->color = BLACK;
    }

    void insert_case2(Node* n)
    {
        return; /* Do nothing since tree is still valid */
    }

    void insert_case3(Node* n)
    {
        parent(n)->color = BLACK;
        uncle(n)->color = BLACK;
        grandparent(n)->color = RED;
        insert_repair_tree(grandparent(n));
    }

    void insert_case4(Node* n)
    {
        Node* p = parent(n);
        Node* g = grandparent(n);

        if (n == p->right && p == g->left) {
            rotate_left(p);
            n = n->left;
        } else if (n == p->left && p == g->right) {
            rotate_right(p);
            n = n->right;
        }

        insert_case4step2(n);
    }

    void insert_case4step2(Node* n)
    {
        Node* p = parent(n);
        Node* g = grandparent(n);

        if (n == p->left) rotate_right(g);
        else rotate_left(g);

        p->color = BLACK;
        g->color = RED;
    }

    void clean_tree(Node* root) {
        if (root != nullptr) {
            clean_tree(root->left);
            clean_tree(root->right);
            delete root;
            deleted++;
        }
    }

    /* Removal */
    void delete_node(Node* n) {
        assert(!n->leaf);
        if (!n->right->leaf && !n->left->leaf) {
            Node* exchange_n = get_largest_descendant(n->left);
        } else {
            delete_one_child(n);
        }
    }

    Node* get_largest_descendant(Node* n) {
        assert(!n->leaf);
        if (n->left->leaf || n->right->leaf) {
            return n;
        } else {
            return get_largest_descendant(n->right);
        }
    }

    void replace_node(Node* n, Node* child){
        child->parent = n->parent;
        if (n == n->parent->left)
            n->parent->left = child;
        else
            n->parent->right = child;
    }

    void delete_one_child(Node* n)
    {
        /*
         * Precondition: n has at most one non-leaf child.
         */
        Node* child;
        Node* abandoned_child;
        if (n->right->leaf) {
            child = n->left;
            abandoned_child = n->right;
        } else {
            child = n->right;
            abandoned_child = n->left;
        }
        assert(abandoned_child->leaf);

        replace_node(n, child);
        if (n->color == BLACK) {
            if (child->color == RED)
                child->color = BLACK;
            else
                delete_case1(child);
        }
        delete n;
        delete abandoned_child;
        deleted+=2;
    }

    void delete_case1(Node* n)
    {
        if (n->parent != nullptr)
            delete_case2(n);
    }

    void delete_case2(Node* n)
    {
        Node* s = sibling(n);

        if (s->color == RED) {
            n->parent->color = RED;
            s->color = BLACK;
            if (n == n->parent->left)
                rotate_left(n->parent);
            else
                rotate_right(n->parent);
        }
        delete_case3(n);
    }

    void delete_case3(Node* n)
    {
        Node* s = sibling(n);

        if ((n->parent->color == BLACK) &&
            (s->color == BLACK) &&
            (s->left->color == BLACK) &&
            (s->right->color == BLACK)) {
            s->color = RED;
            delete_case1(n->parent);
        } else
            delete_case4(n);
    }

    void delete_case4(Node* n)
    {
        Node* s = sibling(n);

        if ((n->parent->color == RED) &&
            (s->color == BLACK) &&
            (s->left->color == BLACK) &&
            (s->right->color == BLACK)) {
            s->color = RED;
            n->parent->color = BLACK;
        } else
            delete_case5(n);
    }

    void delete_case5(Node* n)
    {
        Node* s = sibling(n);

        if  (s->color == BLACK) { /* this if statement is trivial,
            due to case 2 (even though case 2 changed the sibling to a sibling's child,
            the sibling's child can't be red, since no red parent can have a red child). */
            /* the following statements just force the red to be on the left of the left of the parent,
               or right of the right, so case six will rotate correctly. */
            if ((n == n->parent->left) &&
                (s->right->color == BLACK) &&
                (s->left->color == RED)) { /* this last test is trivial too due to cases 2-4. */
                s->color = RED;
                s->left->color = BLACK;
                rotate_right(s);
            } else if ((n == n->parent->right) &&
                       (s->left->color == BLACK) &&
                       (s->right->color == RED)) {/* this last test is trivial too due to cases 2-4. */
                s->color = RED;
                s->right->color = BLACK;
                rotate_left(s);
            }
        }
        delete_case6(n);
    }

    void delete_case6(Node* n)
    {
        Node* s = sibling(n);

        s->color = n->parent->color;
        n->parent->color = BLACK;

        if (n == n->parent->left) {
            s->right->color = BLACK;
            rotate_left(n->parent);
        } else {
            s->left->color = BLACK;
            rotate_right(n->parent);
        }
    }

    void link_nodes() {
        // build double-linked list for all nodes in sorted order by running DFS
        if (root->leaf) return;
        auto minmax = DFS(root);
        max_node = minmax.second;
        min_node = minmax.first;
    }

    std::pair<Node*, Node*> DFS(Node* root) {
        assert(!root->leaf);
        auto res = std::make_pair(root, root);
        if (!root->left->leaf) {
            auto left_and_right = DFS(root->left);
            left_and_right.second->right_sorted = root;
            root->left_sorted = left_and_right.second;
            res.first = left_and_right.first;
        }
        if (!root->right->leaf) {
            auto left_and_right = DFS(root->right);
            left_and_right.first->left_sorted = root;
            root->right_sorted = left_and_right.first;
            res.second = left_and_right.second;
        }
        return res;
    }

public:
    BST() {
        root = new Node();
        created++;
        size = 0;
    }

    ~BST() {
        clean_tree(root);
        assert(created == deleted);
    }

    void update(Node* node) {
        remove(node);
        insert(node);
    }

    Node* insert(double key) {
        Node* n = new Node;
        created++;
        n->key = key;
        n->index = size;
        n->leaf = false;
        insert(n);
        size++;
        return n;
        // todo upd linkage
    }

    void insert(std::vector<double> x) {
        assert(size == 0);
        for (const auto& key : x) {
            Node* n = new Node;
            n->key = key;
            n->index = size;
            n->leaf = false;
            insert(n);
            size++;
        }
        created += x.size();
        link_nodes();
    }

    void remove(Node* n) {
        delete_node(n);
        size--;
    }

    void check_tree_structure() {
        // check all invariants
        if (root == nullptr) assert(size == 0);
        else {
            check_tree_structure_rec(root);

            if (!root->leaf) {
                long left_n = check_linking_left(root);
                long right_n = check_linking_right(root);
                assert(size == left_n + right_n - 1); // one for root counted twice
            }
        }
    }

    int check_tree_structure_rec(Node* root) {
        if (!root->leaf) {
            if (root->color == RED)
                assert((root->left->leaf || root->left->color == BLACK)
                        && (root->right->leaf || root->right->color == BLACK));
            int left_black = 0;
            int right_black = 0;
            left_black = check_tree_structure_rec(root->left);
            if (!root->left->leaf) {
                assert(cmp(root->left, root));
            }
            right_black = check_tree_structure_rec(root->right);
            if (!root->right->leaf) {
                assert(cmp(root, root->right));
            }
            assert(left_black == right_black);
            if (root->color == BLACK) {
                return left_black + 1;
            } else {
                return left_black;
            }
        } else {
            root->color = BLACK;
            assert(root->left == nullptr && root->right == nullptr);
            return 1;
        }
    }

    long check_linking_right(Node* node) {
        assert(!node->leaf);
        if (node->right_sorted != nullptr) {
            assert(cmp(node, node->right_sorted));
            assert(node->left_sorted->right_sorted == node);
            return check_linking_right(node->right_sorted) + 1;
        }
        return 1;
    }

    long check_linking_left(Node* node) {
        assert(!node->leaf);
        if (node->left_sorted != nullptr) {
            assert(cmp(node->left_sorted, node));
            assert(node->right_sorted->left_sorted == node);
            return check_linking_left(node->left_sorted) + 1;
        }
        return 1;
    }

    int get_size() {
        return size;
    }

    std::list<long> get_desc_sorted_indexes() {
        std::list<long> res;
        Node* n = max_node;
        while (n != nullptr) {
            res.push_back(n->index);
            n = n->left_sorted;
        }
        return res;
    }

};

#endif //MACAU_BST_H
