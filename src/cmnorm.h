#ifndef mnorm_cmnorm_H
#define mnorm_cmnorm_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace RcppArmadillo;

List cmnorm(const NumericVector mean,
            const NumericMatrix sigma,
            const NumericVector given_ind,
            const NumericVector given_x,
            NumericVector dependent_ind,
            const bool is_validation,
            const bool is_names,
            Nullable<List> control,
            const int n_cores);

#endif
