#ifndef mnorm_qmnorm_H
#define mnorm_qmnorm_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace RcppArmadillo;

arma::vec qnormFast(arma::vec const &p, 
                    const int mean,
                    const int sd,
                    String method,
                    bool is_validation,
                    const int n_cores);

#endif
