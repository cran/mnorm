#include <RcppArmadillo.h>
#include <omp.h>
#include "cmnorm.h"
using namespace Rcpp;

// [[Rcpp::plugins(openmp)]]
// [[Rcpp::interfaces(r, cpp)]]

//' Parameters of conditional multivariate normal distribution
//' @description This function calculates mean (expectation) and covariance 
//' matrix of conditional multivariate normal distribution.
//' @template param_mean_Template
//' @template param_sigma_Template
//' @template param_given_ind_Template
//' @template param_given_x_Template
//' @template param_dependent_ind_Template
//' @template param_is_validation_Template
//' @template param_is_names_Template
//' @template param_control_Template
//' @template param_n_cores_Template
//' @template details_cmnorm_Template
//' @template return_cmnorm_Template
//' @template example_cmnorm_Template
//' @export
// [[Rcpp::export(rng = false)]]
List cmnorm(const NumericVector mean,
            const NumericMatrix sigma,
            const NumericVector given_ind,
            const NumericVector given_x,
            NumericVector dependent_ind = NumericVector(),
            const bool is_validation = true,
            const bool is_names = true,
            Nullable<List> control = R_NilValue,
            const int n_cores = 1)
{
  // Multiple cores
  omp_set_num_threads(n_cores);
  
  // Deal with control input
  List control1(control);
  bool diff_mean_by_sigma_dg = false;
  if (control !=R_NilValue)
  {
    if (control1.containsElementNamed("diff_mean_by_sigma_dg"))
    {
      diff_mean_by_sigma_dg = control1["diff_mean_by_sigma_dg"];
    }
  }
  
  // Get information concerning initial dimensions
  int n_dim = sigma.ncol();
  
  // Transform mean and sigma to arma
  arma::vec mean_arma = as<arma::vec>(mean);
  arma::mat sigma_arma = as<arma::mat>(sigma);
  
  // Get the number of conditioned components and
  // the number of multivariate normal vectors
  // under consideration
  int n_total = given_x.size();
  int n_given = given_ind.size();
  int n = n_total / n_given;
  int n_dependent = dependent_ind.size();
  
  // Provide input validation if need
  if (is_validation)
  {
    if (n_given >= n_dim)
    {
      std::string stop_message = "At least one element (component) of "
        "multivariate normal vector should be unconditioned. "
        "Please, insure that 'length(given_ind) < length(mean)'.";
      stop(stop_message);
    }
    
    if(n_given == 0)
    {
      std::string stop_message = "At least one element (component) of "
        "multivariate normal vector should be conditioned. "
        "Please, insure that 'length(given_ind) >= 1'.";
      stop(stop_message);
    }
    
    if (n_dim != mean.size())
    {
      std::string stop_message = "Sizes of 'mean' and 'sigma' do not match. "
        "Please, insure that 'length(mean) == ncol(sigma)'.";
      stop(stop_message);
    }
    
    if (is_true(any(given_ind < 1)) | 
        is_true(any(given_ind > n_dim)) |
        is_true(any(is_na(given_ind))))
    {
      std::string stop_message = "Elements out of bounds in 'given_ind'. "
        "Please, insure that "
        "'max(given_ind) <= length(mean)', 'min(given_ind) >= 1' "
        "and 'all(!is.nan(given_ind))'.";
      stop(stop_message);
    }
    
    if (unique(given_ind).size() != n_given)
    {
      std::string stop_message = "Duplicates have been found in 'given_ind'. "
        "Please, insure that 'length(unique(given_ind)) == length(given_ind)'.";
      stop(stop_message);
    }
    
    if (n_dependent > 0)
    {
      if (is_true(any(dependent_ind < 1)) | 
          is_true(any(dependent_ind > n_dim)) |
          is_true(any(is_na(dependent_ind))))
      {
        std::string stop_message = "Elements out of bounds in 'dependent_ind'. "
          "Please, insure that "
          "'max(dependent_ind) <= length(mean)', 'min(dependent_ind) >= 1' "
          "and 'all(!is.nan(dependent_ind))'.";
        stop(stop_message);
      }
    }
    
    if (!sigma_arma.is_sympd())
    {
      std::string stop_message = "Not positively definite covariance matrix. "
        "Please, insure that 'sigma' is positively definite covariance matrix.";
      stop(stop_message);
    }
    
    if (given_x.hasAttribute("dim"))
    {
      NumericVector given_x_dim = given_x.attr("dim");
      if (given_x_dim[1] != n_given)
      {
        std::string stop_message = "Sizes of 'given_x' and 'given_ind' do not "
          "match. Please, insure that 'ncol(given_x) == length(given_ind)'.";
        stop(stop_message);
      }
    }
    else
    {
      if ((given_x.size() % n_given) != 0)
      {
        std::string stop_message = "Size of 'given_x' do not match the "
          "number of conditioned components of multivariate normal vector. "
          "Please, insure that '(length(given_x) %% length(given_ind)) == 0'.";
        stop(stop_message);
      }
    }
    
    if (n_dependent > 0)
    {
      bool is_duplicates = false;
      // Inefficient but clear algorithm for finding
      // duplicate values in given_ind and dependent_ind
      for (int i = 0; i < n_given; i++)
      {
        for (int j = 0; j < n_dependent; j++)
        {
          if (given_ind[i] == dependent_ind[j])
          {
            is_duplicates = true;
            break;
          }
        }
      }
      if (is_duplicates)
      {
        std::string stop_message = "Duplicates have been found in 'given_ind' "
          "and 'dependent_ind'. Every element of multivariate normal vector "
          "should be either conditioned or unconditioned. Please, insure that "
          "'!any(given_ind %in% dependent_ind)'.";
        stop(stop_message);
      }
    }
    
    // Check that the number of cores is correctly specified
    if (n_cores < 1)
    {
      stop("Please, insure that 'n_cores' is a positive integer.");
    }
  }
  
  // Convert vector into matrix preventing
  // global variable being overwritten in R
  NumericVector given_x_vec = as<NumericVector>(clone(given_x));
  if (!given_x.hasAttribute("dim"))
  {
    given_x_vec.attr("dim") = Dimension(n, n_given);
  }
  NumericMatrix given_x_mat = as<NumericMatrix>(given_x_vec);
  
  // Account for empty dependent_ind
  if (n_dependent == 0)
  {
    IntegerVector ind = Rcpp::seq(1, n_dim);
    LogicalVector given_ind_logical = LogicalVector(n_dim);
    given_ind_logical[given_ind - 1] = true;
    dependent_ind = ind[!given_ind_logical];
    n_dependent = dependent_ind.size();
  }
  
  // Subtract one from indexes and transform
  // them into arma format
  arma::uvec dependent_arma = as<arma::uvec>(dependent_ind) - 1;
  arma::uvec given_arma = as<arma::uvec>(given_ind) - 1;
  
  // Transform given_x to arma
  arma::mat X_mat = as<arma::mat>(given_x_mat);
  
  // Get separate parameters for dependent and
  // given (conditioned) components
  arma::rowvec mean_d = mean_arma.elem(dependent_arma).t();
  arma::rowvec mean_g = mean_arma.elem(given_arma).t();
  arma::mat sigma_d = sigma_arma.submat(dependent_arma, dependent_arma);
  arma::mat sigma_g = sigma_arma.submat(given_arma, given_arma);
  arma::mat sigma_dg = sigma_arma.submat(dependent_arma, given_arma);
  
  // Estimate preliminary component
  arma::mat sigma_g_inv = sigma_g.i();
  arma::mat s12s22 = (sigma_dg * sigma_g_inv);
  arma::mat s12s22t = s12s22.t();

  // Estimate conditional mean applying slightly different procedures
  // depending on whether single or multiple cores are used
  arma::mat mean_cond = arma::mat(n, n_dependent);
  arma::mat mat_tmp = (X_mat.each_row() - mean_g) * s12s22t;
  mean_cond = mat_tmp.each_row() + mean_d;
  
  // Estimate conditional covariance
  arma::mat sigma_cond = sigma_d - s12s22t.t() * sigma_dg.t();
  
  // Convert to NumericVector and NumericMatrix
  NumericMatrix mean_cond_numeric = wrap(mean_cond);
  NumericMatrix sigma_cond_numeric = wrap(sigma_cond);

  // Assign names if need
  if (is_names)
  {
    CharacterVector x_names = CharacterVector(n_dependent);
    for (int i = 0; i < n_dependent; i++)
    {
      x_names[i] = "X" + std::to_string((int)dependent_ind[i]);
    }
    colnames(mean_cond_numeric) = x_names;
    colnames(sigma_cond_numeric) = x_names;
    rownames(sigma_cond_numeric) = x_names;
  }
  
  // Aggregate the results into the list
  List return_list = List::create(Named("mean") = mean_cond_numeric,
                                  Named("sigma") = sigma_cond_numeric,
                                  Named("s12s22") = s12s22,
                                  Named("sigma_d") = sigma_d,
                                  Named("sigma_g") = sigma_g,
                                  Named("sigma_dg") = sigma_dg,
                                  Named("sigma_g_inv") = sigma_g_inv);
  
  // Calculate derivatives of conditional mean
  // respect to sigma_dg
  if (diff_mean_by_sigma_dg)
  {
    arma::mat diff_mean_by_sigma_dg_mat = arma::mat(n, n_given);
    for (int i = 0; i < n; i++)
    {
      diff_mean_by_sigma_dg_mat.row(i) = (X_mat.row(i) - mean_g) * 
                                         sigma_g_inv;
    }
    return_list["diff_mean_by_sigma"] = diff_mean_by_sigma_dg_mat;
  }
  
  // Set the class
  return_list.attr("class") = "mnorm_cmnorm";
  
  return (return_list);
}
