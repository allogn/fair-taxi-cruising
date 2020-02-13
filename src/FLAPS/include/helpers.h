//
// Created by Alvis Logins on 2019-03-14.
//

#ifndef MACAU_HELPERS_H
#define MACAU_HELPERS_H

#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

    return idx;
}

inline double sgn(double x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}

#endif //MACAU_HELPERS_H
