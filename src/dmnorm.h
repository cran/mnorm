#ifndef mnorm_dmnorm_H
#define mnorm_dmnorm_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace RcppArmadillo;

List dmnorm(const NumericVector x,
            const NumericVector mean,
            const NumericMatrix sigma,
            const NumericVector given_ind,
            const bool log,
            const bool grad_x,
            const bool grad_sigma,
            const bool is_validation,
            const Nullable<List> control,
            const int n_cores);

#endif
