#ifndef mnorm_pmnorm_H
#define mnorm_pmnorm_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace RcppArmadillo;

List pmnorm(const NumericVector lower, const NumericVector upper,
            const NumericVector given_x,
            const NumericVector mean, const NumericMatrix sigma,
            const NumericVector given_ind,
            const int n_sim,
            const String method,
            const String ordering,
            const bool log,
            const bool grad_lower,
            const bool grad_upper,
            const bool grad_sigma,
            const bool grad_given,
            const bool is_validation,
            Nullable<List> control,
            const int n_cores,
            Nullable<List> marginal,
            const bool grad_marginal,
            const bool grad_marginal_prob);
  
double GHK(const NumericVector lower, 
           const NumericVector upper, 
           const NumericMatrix sigma, 
           const arma::mat h,
           const String ordering,
           const int n_sim,
           const int n_cores);

#endif
