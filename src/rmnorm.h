#ifndef mnorm_rmnorm_H
#define mnorm_rmnorm_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace RcppArmadillo;

NumericMatrix rmnorm(const int n,
                     const NumericVector mean,
                     const NumericMatrix sigma,
                     const NumericVector given_ind,
                     const NumericVector given_x,
                     const NumericVector dependent_ind,
                     const bool is_validation,
                     const int n_cores);

#endif
