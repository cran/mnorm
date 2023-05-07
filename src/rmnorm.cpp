#define ARMA_DONT_USE_OPENMP
#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "rmnorm.h"
#include "cmnorm.h"
using namespace Rcpp;

#ifdef _OPENMP
// [[Rcpp::plugins(openmp)]]
#endif
// [[Rcpp::interfaces(r, cpp)]]

//' Random number generator for (conditional) multivariate normal distribution
//' @description This function generates random numbers (i.e. variates) from 
//' (conditional) multivariate normal distribution.
//' @param n positive integer representing the number of random variates
//' to be generated from (conditional) multivariate normal distribution.
//' If \code{given_ind} is not empty vector then \code{n} should be
//' be equal to \code{nrow(given_x)}.
//' @template param_mean_Template
//' @template param_sigma_Template
//' @template param_given_ind_Template
//' @template param_given_x_Template
//' @template param_dependent_ind_Template
//' @template param_is_validation_Template
//' @template param_n_cores_Template
//' @details This function uses Cholesky decomposition to generate multivariate
//' normal variates from independent standard normal variates.
//' @template example_rmnorm_Template
//' @return This function returns a numeric matrix which rows a random variates
//' from (conditional) multivariate normal distribution with mean equal to
//' \code{mean} and covariance equal to \code{sigma}. If \code{given_x} and 
//' \code{given_ind} are also provided then random variates will be from
//' conditional multivariate normal distribution. Please, see details section
//' of \code{\link[mnorm]{cmnorm}} to get additional information on the 
//' conditioning procedure.
//' @export
// [[Rcpp::export(rng = true)]]
NumericMatrix rmnorm(const int n,
                     const NumericVector mean,
                     const NumericMatrix sigma,
                     const NumericVector given_ind = NumericVector(),
                     const NumericVector given_x = NumericVector(),
                     const NumericVector dependent_ind = NumericVector(),
                     const bool is_validation = true,
                     const int n_cores = 1)
{
  if (given_ind.size() == 0)
  {
    const int n_dim = mean.size();
    
    const arma::rowvec mean_arma = as<arma::rowvec>(mean);
    const arma::mat sigma_arma = as<arma::mat>(sigma);
    const arma::mat L = arma::chol(sigma_arma, "lower");
    
    arma::mat z = arma::randn(n_dim, n);
    arma::mat x = L * z;
    x = x.t();
    x = x.each_row() + mean_arma;
    
    return(wrap(x));
  }
  
  if ((given_x.size() / given_ind.size()) != n)
  {
    stop("Please, insure that 'n' equals to 'nrow(given_x)'.");
  }
  List cond = cmnorm(mean, sigma, 
                     given_ind, given_x, dependent_ind,
                     is_validation, false, R_NilValue, n_cores);
  const arma::mat mean_c = cond["mean"];
  const arma::mat sigma_c = cond["sigma"];
  const arma::mat L_c = arma::chol(sigma_c, "lower");
    
  const int n_dim = sigma_c.n_rows;
  arma::mat z = arma::randn(n_dim, n);
  arma::mat x = L_c * z;
  x = x.t() + mean_c;
    
  return(wrap(x));
}
