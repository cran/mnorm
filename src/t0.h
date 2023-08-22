#ifndef mnorm_t0_H
#define mnorm_t0_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace RcppArmadillo;

List pbetaDiff(const arma::vec x, const double p, const double q, const int n, 
               const bool is_validation, const Nullable<List> control);

List dt0(const arma::vec x, const double df, const bool log,
         const bool grad_x, const bool grad_df);
  
List pt0(const arma::vec x, const double df, const bool log,
         const bool grad_x, const bool grad_df, const int n);

NumericVector rt0(const int n, const double df);

NumericVector qt0(const NumericVector x, const double df);

#endif
