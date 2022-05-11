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
            const int n_cores);

arma::vec pmnorm2(const arma::vec x1,
                  const arma::vec x2,
                  const arma::vec x,
                  const arma::vec adj,
                  const arma::vec adj1,
                  const arma::vec adj2,
                  const int n_cores);
  
double GHK(const NumericVector lower, 
           const NumericVector upper, 
           const NumericMatrix sigma, 
           const arma::mat h,
           const String ordering,
           const int n_sim,
           const int n_cores);

#endif
