//
// Created by Alvis Logins on 2019-04-19.
//

#ifndef MACAU_NETWORKPARAMS_H
#define MACAU_NETWORKPARAMS_H

struct NetworkParams {
    NetworkParams() {
        m = 0;
        n = 0;
    }
    virtual ~NetworkParams() = default;
    unsigned long m;
    unsigned long n;
};

#endif //MACAU_NETWORKPARAMS_H
