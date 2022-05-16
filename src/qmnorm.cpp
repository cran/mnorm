#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "cmnorm.h"
#include "dmnorm.h"
#include "qmnorm.h"
using namespace Rcpp;

#ifdef _OPENMP
// [[Rcpp::plugins(openmp)]]
#endif
// [[Rcpp::interfaces(r, cpp)]]

// --------------------------------------
// --------------------------------------
// --------------------------------------

//' Quantile function of a normal distribution
//' @description Calculate quantile of a normal distribution using
//' one of the available methods.
//' @param p numeric vector of values between 0 and 1 representing levels of
//' the quantiles.
//' @param mean numeric value representing the expectation of a
//' normal distribution.
//' @param sd positive numeric value representing standard deviation of a
//' normal distribution.
//' @param method character representing the method to be used for
//' quantile calculation. Available options are "Voutier" (default) and "Shore".
//' @template param_is_validation_Template
//' @template param_n_cores_Template
//' @details If \code{method = "Voutier"} then the method of P. Voutier (2010)
//' is used which maximum absolute error is about \eqn{0.000025}.
//' If \code{method = "Shore"} then the approach proposed
//' by H. Shore (1982) is applied which maximum absolute error is about
//' \eqn{0.026} for quantiles of level between \eqn{0.0001} 
//' and \eqn{0.9999}.
//' @return The function returns a vector of \code{p}-level quantiles of a
//' normal distribution with mean equal to \code{mean} and standard 
//' deviation equal to \code{sd}.
//' @references H. Shore (1982) <doi:10.2307/2347972>
//' @references P. Voutier (2010) <doi:10.48550/arXiv.1002.0567>
//' @examples qnormFast(c(0.1, 0.9), mean = 1, sd = 2)
// [[Rcpp::export(rng = false)]]
arma::vec qnormFast(arma::vec const &p, 
                    const int mean = 0,
                    const int sd = 1,
                    String method = "Voutier",
                    bool is_validation = true,
                    const int n_cores = 1)
{
  // The number of quantiles to calculate
  const int n = p.size();
  
  // Validation
  if (is_validation)
  {
    // Check that all levels are within (0, 1) interval
    if (any(p >= 1) || any(p <= 0))
    {
      std::string stop_message = "Some values of 'p' are not between 0 and 1. "
      "Please, insure that 'all((p < 1) & (p > 0))'.";
      stop(stop_message);
    }
    
    // Check sd value
    if (sd <= 0)
    {
      stop("Please, insure that 'sd' is positive.");
    }
    
    // Check method
    if ((method != "Voutier") && (method != "Shore"))
    {
      stop("Please, insure that 'method' value is correct.");
    }
    
    // Check that the number of cores is correctly specified
    if (n_cores < 1)
    {
      stop("Please, insure that 'n_cores' is a positive integer.");
    }
  }
  
  // The vector of quantiles (output)
  arma::vec val(n);
  
  if (method == "Voutier")
  {
    // Voutier, P.M. (2010). "A New Approximation to the Normal 
    // Distribution Quantile Function". arXiv: Computation.
    
    // Constants for the central region
    const double a0 = 0.195740115269792;
    const double a1 = -0.652871358365296;
    const double a2 = 1.246899760652504;
    const double b0 = 0.155331081623168;
    const double b1 = -0.839293158122257;
      
    // Constants for the tails
    const double c0 = 16.682320830719986527;
    const double c1 = 4.120411523939115059;
    const double c2 = 0.029814187308200211;
    const double c3 = -1.000182518730158122;
    const double d0 = 7.173787663925508066;
    const double d1 = 8.759693508958633869;
    
    // Routine
    double r;
    double q;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_cores) private(r, q) if (n_cores > 1)
    #endif
    for (int i = 0; i < n; i++)
    {
      if ((0.025 <= p.at(i)) && (p.at(i) <= 0.975))
      {
        q = p.at(i) - 0.5;
        r = std::pow(q, 2.0);
        val.at(i) = q * (a2 + (a1 * r + a0) / 
                        (std::pow(r, 2.0) + b1 * r + b0));
      }
      else
      {
        if (p.at(i) < 0.5)
        {
          r = sqrt(log(1 / (std::pow(p.at(i), 2.0))));
        }
        else
        {
          r = sqrt(log(1 / (std::pow(1 - p.at(i), 2.0))));
        }
        val.at(i) = c3 * r + c2 + 
                    (c1 * r + c0) / (std::pow(r, 2.0) + d1 * r + d0);
        if (p.at(i) > 0.5)
        {
          val.at(i) = -val.at(i);
        }
      }
    }
  }
  
  if (method == "Shore")
  {
    // Shore, H (1982). "Simple Approximations for the Inverse Cumulative 
    // Function, the Density Function and the Loss Integral of the Normal
    // Distribution". Journal of the Royal Statistical Society. 
    // Series C (Applied Statistics). 31 (2): 108–114.
    
    // Routine
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_cores) if (n_cores > 1)
    #endif
    for (int i = 0; i < n; i++)
    {
      if (p.at(i) >= 0.5)
      {
        val.at(i) = 5.5556 * (1 - pow((1 - p.at(i)) / p.at(i), 0.1186));
      }
      else
      {
        val.at(i) = -5.5556 * (1 - pow(p.at(i) / (1 - p.at(i)), 0.1186));
      }
    }
  }
  
  // Adjust for non-identity variance
  if (sd != 1)
  {
    val = val * sd;
  }
  
  // Adjust for non-zero mean
  if (mean != 0)
  {
    val = val + mean;
  }
  
  return(val);
}
