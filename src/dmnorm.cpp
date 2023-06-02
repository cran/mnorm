#define ARMA_DONT_USE_OPENMP
#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "cmnorm.h"
#include "dmnorm.h"
using namespace Rcpp;

#ifdef _OPENMP
// [[Rcpp::plugins(openmp)]]
#endif
// [[Rcpp::interfaces(r, cpp)]]

// --------------------------------------
// --------------------------------------
// --------------------------------------


//' Density of (conditional) multivariate normal distribution
//' @description This function calculates and differentiates density of 
//' (conditional) multivariate normal distribution.
//' @template param_x_Template
//' @template param_mean_Template
//' @template param_sigma_Template
//' @template param_given_ind_2_Template
//' @template param_log_Template
//' @template param_grad_x_Template
//' @template param_grad_sigma_Template
//' @template param_is_validation_Template
//' @template param_control_Template
//' @template param_n_cores_Template
//' @template details_dmnorm_Template
//' @template return_dmnorm_Template
//' @template example_dmnorm_Template
//' @references E. Kossova., B. Potanin (2018). 
//' Heckman method and switching regression model multivariate generalization.
//' Applied Econometrics, vol. 50, pages 114-143.
//' @export
// [[Rcpp::export(rng = false)]]
List dmnorm(const NumericVector x,
            const NumericVector mean,
            const NumericMatrix sigma,
            const NumericVector given_ind = NumericVector(),
            const bool log = false,
            const bool grad_x = false,
            const bool grad_sigma = false,
            const bool is_validation = true,
            const Nullable<List> control = R_NilValue,
            const int n_cores = 1)
{
  // Create output list
  List return_list;
  
  // Check whether any gradients should be calculated
  const bool is_grad = (grad_x || grad_sigma);

  // Get number of dimensions
  const int n_dim = sigma.nrow();
  
  // Get total number of coordinates
  const int n_total = x.size();
  
  // Get number of observations
  const int n = n_total / n_dim;
  
  // Get the number of conditioned components
  int n_given = given_ind.size();
  
  // Deal with control input
  List control1(control);
  LogicalVector is_use;
  int is_use_n = n;
  if (control != R_NilValue)
  {
    if (control1.containsElementNamed("is_use"))
    {
      is_use = control1["is_use"];
      is_use_n = sum(is_use);
    }
  }
  
  // Provide input validation if need
  if (is_validation)
  {
    int mean_size_tmp = mean.size();
    if (n_dim != mean_size_tmp)
    {
      std::string stop_message = "Sizes of 'mean' and 'sigma' do not match. "
      "Please, insure that 'length(mean) == ncol(sigma)'.";
      stop(stop_message);
    }
    
    if (n_given > 0)
    {
      if (is_true(any(given_ind < 1)) || 
          is_true(any(given_ind > n_dim)) ||
          is_true(any(is_na(given_ind))))
      {
        std::string stop_message = "Elements out of bounds in 'given_ind'. "
        "Please, insure that "
        "'max(given_ind) <= length(mean)', 'min(given_ind) >= 1' "
        "and 'all(!is.nan(given_ind)).'";
        stop(stop_message);
      }
    
      int unique_given_ind_size_tmp = unique(given_ind).size();
      int given_ind_size_tmp = given_ind.size();
      if (unique_given_ind_size_tmp != given_ind_size_tmp)
      {
        std::string stop_message = "Duplicates have been found in 'given_ind'. "
        "Please, insure that 'length(unique(given_ind)) == length(given_ind)'.";
        stop(stop_message);
      }
    }
    
    if (!as<arma::mat>(sigma).is_sympd())
    {
      std::string stop_message = "Not positively definite covariance matrix. "
      "Please, insure that 'sigma' is positively definite covariance matrix.";
      stop(stop_message);
    }
    
    // Check that the number of cores is correctly specified
    if (n_cores < 1)
    {
      stop("Please, insure that 'n_cores' is a positive integer.");
    }
  }
  
  // Convert vector of arguments 
  // into a matrix if need
  NumericVector x_vec = as<NumericVector>(clone(x));
  if (!x.hasAttribute("dim"))
  {
    x_vec.attr("dim") = Dimension(n, n_dim);
  }
  NumericMatrix x_mat = as<NumericMatrix>(x_vec);
  
  // Deal with calculations only for particular observations
  if (is_use_n != n)
  {
    if (is_use_n == 0)
    {
      return_list["den"] = NumericVector(n);
      return_list.attr("class") = "mnorm_dmnorm";
      return(return_list);
    }
    
    NumericMatrix x_use(is_use_n, n_dim);
    int counter_use = 0;
    for (int i = 0; i < n; i++)
    {
      if (is_use[i])
      {
        x_use(counter_use, _) = x_mat(i, _);
        counter_use++;
      }
    }

    List return_list_use = dmnorm(x_use,
                                  mean, sigma,
                                  given_ind, log,
                                  false, false, false, 
                                  R_NilValue, n_cores);
    NumericVector den_use = return_list_use["den"];
    NumericVector den_new(n);
    counter_use = 0;
    for (int i = 0; i < n; i++)
    {
      if (is_use[i])
      {
        den_new[i] = den_use[counter_use];
        counter_use++;
      }
    }
    return_list["den"] = den_new;
    return_list.attr("class") = "mnorm_dmnorm";
    return(return_list);
  }
  
  // Adjust for zero mean
  if (mean.size() != 0)
  {
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n_dim; j++)
      {
        x_mat(i, j) = x_mat(i, j) - mean[j];
      }
    }
  }
  
  // Create indexes of dependent variables
  NumericVector dependent_ind;
  IntegerVector ind = Rcpp::seq(1, n_dim);
  LogicalVector given_ind_logical = LogicalVector(n_dim);
  if (n_given > 0)
  {
    given_ind_logical[given_ind - 1] = true;
    dependent_ind = ind[!given_ind_logical];
  }
  else
  {
    dependent_ind = ind;
  }
  int n_dependent = dependent_ind.size();
  
  // Account for conditioning
  NumericMatrix mean_cond;
  NumericMatrix sigma_cond;
  NumericMatrix x_g;
  NumericMatrix x_d;
  arma::mat s12s22;
  arma::mat diff_mean_by_sigma;
  List cond;
  if (n_given > 0)
  {
    // Create a matrix of conditioned (given) values
    x_g = NumericMatrix(n, n_given);
    for(int i = 0; i < n_given; i++)
    {
      x_g(_, i) = x_mat(_, given_ind[i] - 1);
    }

    // Get conditional distribution parameters
    NumericVector mean_zero = NumericVector(n_dim);
    List cmnorm_control = List::create(
      Named("diff_mean_by_sigma_dg") = grad_sigma);
    cond = cmnorm(mean_zero, sigma,
                  given_ind, x_g,
                  NumericVector(),
                  false, false, cmnorm_control, n_cores);
    NumericMatrix mean_cond_tmp = cond["mean"];
    NumericMatrix sigma_cond_tmp = cond["sigma"];
    arma::mat s12s22_tmp = cond["s12s22"];
    mean_cond = mean_cond_tmp;
    sigma_cond = sigma_cond_tmp;
    s12s22 = s12s22_tmp;
    if (grad_sigma)
    {
      arma::mat diff_mean_by_sigma_tmp = cond["diff_mean_by_sigma"];
      diff_mean_by_sigma = diff_mean_by_sigma_tmp;
    }

    // Create matrix of new points adjusted for conditioned mean
    x_d = NumericMatrix(n, n_dependent);
    for (int i = 0; i < n_dependent; i++)
    {
      x_d(_, i) = x_mat(_, dependent_ind[i] - 1) - mean_cond(_, i);
    }
  }
  else
  {
    sigma_cond = sigma;
    x_d = x_mat;
  }
  
  // Transform to arma
  arma::mat const x_d_arma(x_d.begin(), n, n_dependent, false);
  arma::mat const sigma_cond_arma(sigma_cond.begin(), 
                                  n_dependent, n_dependent, false);
  arma::mat const L_cond = chol(sigma_cond_arma, "upper");
  arma::mat const L_cond_inv = trimatu(L_cond).i();

  // Estimate logarithm of the density
  arma::vec den_log = arma::vec(n);
  double const det_adj_val = -0.5 * std::log(arma::det(2 * M_PI * 
                                                       sigma_cond_arma));
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(n_cores) if (n_cores > 1)
  #endif
  for (int i = 0; i < n; i++)
  {
    // Account for triangularity of
    // Cholesky decomposition's inverse
    double dot_prod(0.);
    for(int j1 = 0; j1 < n_dependent; j1++)
    {
      double row_sum(0.);
      for(int j2 = 0; j2 <= j1; j2++)
      {
        row_sum += x_d_arma.at(i, j2) * L_cond_inv.at(j2, j1);
      }
      dot_prod += row_sum * row_sum;
    }
    den_log.at(i) = det_adj_val - 0.5 * dot_prod;
  }
  
  // Estimate the density itself if need
  arma::vec den;
  if(!log || (is_grad))
  {
    den = arma::vec(n);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_cores) if (n_cores > 1)
    #endif
    for (int i = 0; i < n; i++)
    {
      den.at(i) = std::exp(den_log.at(i));
    }
    if(!log)
    {
      return_list["den"] = den;
    }
  }
  if (log)
  {
    return_list["den"] = den_log;
  }
  
  // If no gradients are need then return the results
  if (!is_grad)
  {
    return_list.attr("class") = "mnorm_dmnorm";
    return(return_list);
  }
  
  // Subtract one from indexes and transform
  // them into arma format
  arma::uvec dependent_arma = as<arma::uvec>(dependent_ind) - 1;
  arma::uvec given_arma;
  if (n_given > 0)
  {
    given_arma = as<arma::uvec>(given_ind) - 1;
  }

  // Calculate gradient respect to the argument
    // respect to the arguments of dependent elements
  arma::mat grad_x_arma = arma::mat(n, n_dim);
  arma::mat const sigma_cond_inv = sigma_cond_arma.i();
  grad_x_arma.cols(dependent_arma) = x_d_arma * (-sigma_cond_inv);
    // respect to the arguments of given (conditioned) elements
  arma::mat const x_g_arma(x_g.begin(), n, n_given, false);
  if (n_given > 0)
  {
    grad_x_arma.cols(given_arma) = grad_x_arma.cols(dependent_arma) * (-s12s22);
  }

  // Vectors to help convert indexes of multivariate normal
  // vector to the ordered indexes of dependent and given components
  NumericVector ind_to_d = NumericVector(n_dim);
  NumericVector ind_to_g = NumericVector(n_dim);
  NumericVector d_to_ind = NumericVector(n_dependent);
  NumericVector g_to_ind = NumericVector(n_given);
  if (grad_sigma)
  {
    int counter_d = 0;
    int counter_g = 0;
    for (int i = 0; i < n_dim; i++)
    {
      if (given_ind_logical[i])
      {
        ind_to_g[i] = counter_g;
        g_to_ind[counter_g] = i;
        counter_g++;
      }
      else
      {
        ind_to_d[i] = counter_d;
        d_to_ind[counter_d] = i;
        counter_d++;
      }
    }
  }
  
  // Calculate a gradient respect to the covariance matrix
  arma::cube grad_sigma_arma;
  if (grad_sigma)
  {
    grad_sigma_arma = arma::cube(n_dim, n_dim, n);
    // respect to sigma_d elements
    for (int i = 0; i < n_dependent; i++)
    {
      for (int j = i; j < n_dependent; j++)
      {
        if (i != j)
        {
          grad_sigma_arma.tube(d_to_ind[i], d_to_ind[j]) = 
            grad_x_arma.col(d_to_ind[i]) % 
            grad_x_arma.col(d_to_ind[j]) -
            sigma_cond_inv(i, j);
          grad_sigma_arma.tube(d_to_ind[j], d_to_ind[i]) = 
            grad_sigma_arma.tube(d_to_ind[i], d_to_ind[j]);
        }
        else
        {
          grad_sigma_arma.tube(d_to_ind[i], d_to_ind[j]) = 
            (pow(grad_x_arma.col(d_to_ind[i]), 2) -
            sigma_cond_inv(i, j)) / 2;
        }
      }
    }
    // respect to sigma_dg and sigma_gg elements
    // if there is some conditioning
    if (n_given > 0)
    {
      diff_mean_by_sigma = -diff_mean_by_sigma;
      // respect to sigma_dg elements
      for (int i = 0; i < n_dependent; i++)
      {
        for (int j = 0; j < n_given; j++)
        {
          // part associated with conditional mean
          grad_sigma_arma.tube(d_to_ind[i], g_to_ind[j]) =
            grad_x_arma.col(d_to_ind[i]) % diff_mean_by_sigma.col(j);
          for (int j1 = 0; j1 < n_dependent; j1++)
          {
            // part associated with conditional covariance
            grad_sigma_arma.tube(d_to_ind[i], g_to_ind[j]) =
              grad_sigma_arma.tube(d_to_ind[i], g_to_ind[j]) -
              (1 + (i == j1)) * s12s22.at(j1, j) *
              grad_sigma_arma.tube(d_to_ind[i], d_to_ind[j1]);
          }
          grad_sigma_arma.tube(g_to_ind[j], d_to_ind[i]) =
            grad_sigma_arma.tube(d_to_ind[i], g_to_ind[j]);
        }
      }
      // respect to sigma_g elements
      arma::mat sigma_dg = cond["sigma_dg"];
      arma::mat sigma_g_inv = cond["sigma_g_inv"];
      for (int i = 0; i < n_given; i++)
      {
        for (int j = i; j < n_given; j++)
        {
          arma::mat I_g = arma::mat(n_given, n_given, arma::fill::zeros);
          I_g.at(i, j) = 1;
          I_g.at(j, i) = 1;
          arma::mat mat_tmp = sigma_dg * sigma_g_inv * I_g * 
                              sigma_g_inv * sigma_dg.t();
          arma::mat mat_tmp2 = x_g_arma * 
                               (sigma_dg * sigma_g_inv * I_g * sigma_g_inv).t();
          // part associated with conditional mean
          arma::mat mat_tmp3 = sum(mat_tmp2 % 
                                   grad_x_arma.cols(dependent_arma), 1);
          for (int i1 = 0; i1 < n_dependent; i1++)
          {
            for (int j1 = i1; j1 < n_dependent; j1++)
            {
              // part associated with conditional covariance
              arma::colvec mat_tmp4 = grad_sigma_arma.tube(d_to_ind[i1], 
                                                           d_to_ind[j1]);
              mat_tmp3 = mat_tmp3 + mat_tmp.at(i1, j1) * mat_tmp4;
            }
          }
          grad_sigma_arma.tube(g_to_ind[i], g_to_ind[j]) = mat_tmp3;
          grad_sigma_arma.tube(g_to_ind[j], g_to_ind[i]) = mat_tmp3;
        }
      }
    }
  }

  // Add a gradient respect to Cholesky in future
  // Formula: sigma^(-1) %*% t(x - mu) %*% (x - mu) %*% L^(-T) - L^(-T)
  
  // Deal with logarithm if need (after all derivatives have been estimated)
  if (!log)
  {
    grad_x_arma = grad_x_arma.each_col() % den;
    if (grad_sigma)
    {
      for (int i = 0; i < n; i++)
      {
        grad_sigma_arma.slice(i) *= den.at(i);
      }
    }
  }
  
  // Aggregate the result into the output (return) list
  return_list["grad_x"] = grad_x_arma;
  if (grad_sigma)
  {
    return_list["grad_sigma"] = grad_sigma_arma;
  }
  
  // Return the results
  return_list.attr("class") = "mnorm_dmnorm";
  return(return_list);
}
