#define ARMA_DONT_USE_OPENMP
#include <RcppArmadillo.h>
#include <hpa.h>
#include "halton.h"
#include "pmnorm.h"
#include "dmnorm.h"
#include "GaussLegendre.h"
#include "cmnorm.h"
#include "qmnorm.h"
#include "t0.h"
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

#ifdef _OPENMP
// [[Rcpp::plugins(openmp)]]
#endif
// [[Rcpp::interfaces(r, cpp)]]

// --------------------------------------
// --------------------------------------
// --------------------------------------

//' Probabilities of (conditional) multivariate normal distribution
//' @description This function calculates and differentiates probabilities of
//' (conditional) multivariate normal distribution.
//' @template details_pmnorm_Template
//' @template param_lower_Template
//' @template param_upper_Template
//' @template param_given_x_Template
//' @template param_mean_Template
//' @template param_sigma_Template
//' @template param_given_ind_Template
//' @template param_n_sim_Template
//' @template param_method_Template
//' @template param_ordering_Template
//' @template param_log_Template
//' @template param_grad_lower_Template
//' @template param_grad_upper_Template
//' @template param_grad_sigma_pmnorm_Template
//' @template param_grad_given_Template
//' @template param_is_validation_Template
//' @template param_control_Template
//' @template param_n_cores_Template
//' @template param_marginal_Template
//' @template param_grad_marginal_Template
//' @template param_grad_marginal_prob_Template
//' @template return_pmnorm_Template
//' @template example_pmnorm_Template
//' @references Genz, A. (2004), Numerical computation of rectangular bivariate 
//' and trivariate normal and t-probabilities, Statistics and 
//' Computing, 14, 251-260.
//' @references Genz, A. and Bretz, F. (2009), Computation of Multivariate 
//' Normal and t Probabilities. Lecture Notes in Statistics, Vol. 195. 
//' Springer-Verlag, Heidelberg.
//' @references E. Kossova, B. Potanin (2018). 
//' Heckman method and switching regression model multivariate generalization.
//' Applied Econometrics, vol. 50, pages 114-143.
//' @references H. I. Gassmann (2003). 
//' Multivariate Normal Probabilities: Implementing an Old Idea of Plackett's.
//' Journal of Computational and Graphical Statistics, vol. 12 (3),
//' pages 731-752.
//' @export
// [[Rcpp::export(rng = false)]]
List pmnorm(const NumericVector lower, 
            const NumericVector upper,
            const NumericVector given_x = NumericVector(),
            const NumericVector mean = NumericVector(), 
            const NumericMatrix sigma = NumericMatrix(),
            const NumericVector given_ind = NumericVector(),
            const int n_sim = 1000,
            const String method = "default",
            const String ordering = "mean",
            const bool log = false,
            const bool grad_lower = false,
            const bool grad_upper = false,
            const bool grad_sigma = false,
            const bool grad_given = false,
            const bool is_validation = true,
            const Nullable<List> control = R_NilValue,
            const int n_cores = 1,
            const Nullable<List> marginal = R_NilValue,
            const bool grad_marginal = false,
            const bool grad_marginal_prob = false)
{
  // Create output list
  List return_list;
  
  // Check whether any gradients should be calculated
  const bool is_grad = (grad_lower || grad_upper || grad_sigma || 
                        grad_given || grad_marginal);
  
  // Get number of dimensions
  const int n_dim = sigma.nrow();
  
  // Get the number of conditioned and unconditioned components
  const int n_given = given_ind.size();
  const int n_dependent = n_dim - n_given;
  
  // Get number of observations
  const int n = lower.size() / n_dependent;
  
  // Get the size of marginal distributions list
  List marginal_par(marginal);
  int n_marginal = 0;
  if (marginal != R_NilValue)
  {
    n_marginal = marginal_par.size();
  }
  const bool is_marginal = n_marginal > 0;
  
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
    
    if(n_given > 0)
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
    
    if ((ordering != "NO") && (ordering != "mean") && (ordering != "variance"))
    {
      std::string stop_message = "Incorrect ordering method has been provided. "
      "Please, insure that 'ordering' input argument value is correct.";
      stop(stop_message);
    }
    
    // Check that the number of cores is correctly specified
    if (n_cores < 1)
    {
      stop("Please, insure that 'n_cores' is a positive integer.");
    }
    
    // Check correctness of marginal gradients
    if (grad_marginal_prob)
    {
      if (!grad_marginal)
      {
        std::string stop_message = "If 'grad_marginal_prob' is 'TRUE' "
        "then 'grad_marginal' also should be 'TRUE'.";
        stop(stop_message);
      }
    }
  }
  
  // Deal with control input
  List control1(control);
  LogicalVector is_use;
  int is_use_n = n;
  if (control !=R_NilValue)
  {
    if (control1.containsElementNamed("is_use"))
    {
      is_use = control1["is_use"];
      is_use_n = sum(is_use);
    }
  }
  
  // Get marginal distributions parameters
  NumericVector n_marginal_par;
  CharacterVector marginal_names;
  if (is_marginal)
  {
    marginal_names = marginal_par.names();
    n_marginal_par = NumericVector(n_dim);
    if ((n_marginal != n_dim) | (marginal_names.size() != n_dim))
    {
      std::string stop_message = "Wrong size of 'marginal' argument. "
      "Please, insure that it's length coincide with the number of "
      "multivariate normal distribution dimensions "
      "(including conditioned elements)";
      stop(stop_message);
    }
  }

  // Convert vector of lower and upper arguments
  // as well as a vector of conditioned value into 
  // a matrix if need
    // lower
  NumericVector lower_vec = as<NumericVector>(clone(lower));
  if (!lower.hasAttribute("dim"))
  {
    lower_vec.attr("dim") = Dimension(n, n_dependent);
  }
  NumericMatrix lower_mat = as<NumericMatrix>(lower_vec);

    // upper
  NumericVector upper_vec = as<NumericVector>(clone(upper));
  if (!upper.hasAttribute("dim"))
  {
    upper_vec.attr("dim") = Dimension(n, n_dependent);
  }
  NumericMatrix upper_mat = as<NumericMatrix>(upper_vec);
    // given
  NumericVector given_x_vec = as<NumericVector>(clone(given_x));
  if (!given_x.hasAttribute("dim"))
  {
    given_x_vec.attr("dim") = Dimension(n, n_given);
  }
  NumericMatrix given_x_mat = as<NumericMatrix>(given_x_vec);

  // Deal with calculations only for particular observations
  if (is_use_n != n)
  {
    if (is_use_n == 0)
    {
      return_list["prob"] = NumericVector(n);
      return_list.attr("class") = "mnorm_pmnorm";
      return(return_list);
    }
    
    NumericMatrix lower_use(is_use_n, n_dependent);
    NumericMatrix upper_use(is_use_n, n_dependent);
    NumericMatrix given_x_use(is_use_n, n_given);
    int counter_use = 0;
    for (int i = 0; i < n; i++)
    {
      if (is_use[i])
      {
        lower_use(counter_use, _) = lower_mat(i, _);
        upper_use(counter_use, _) = upper_mat(i, _);
        given_x_use(counter_use, _) = given_x_mat(i, _);
        counter_use++;
      }
    }

    List return_list_use = pmnorm(lower_use, upper_use,
                                  given_x_use,
                                  mean, sigma,
                                  given_ind,
                                  n_sim, method, ordering, log,
                                  false, false, false, false,
                                  false, 
                                  R_NilValue, n_cores, marginal);
    NumericVector prob_use = return_list_use["prob"];
    NumericVector prob_new(n);
    counter_use = 0;
    for (int i = 0; i < n; i++)
    {
      if (is_use[i])
      {
        prob_new[i] = prob_use[counter_use];
        counter_use++;
      }
    }
    return_list["prob"] = prob_new;
    return_list.attr("class") = "mnorm_pmnorm";
    return(return_list);
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
  
  // Vectors to help convert indexes of multivariate normal
  // vector to the ordered indexes of dependent and given components
  NumericVector ind_to_d = NumericVector(n_dim);
  NumericVector ind_to_g = NumericVector(n_dim);
  NumericVector d_to_ind = NumericVector(n_dependent);
  NumericVector g_to_ind = NumericVector(n_given);
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

  // Adjust for zero mean
  if (mean.size() != 0)
  {
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n_dependent; j++)
      {
        int j_d = d_to_ind(j);
        lower_mat(i, j) = lower_mat(i, j) - mean[j_d];
        upper_mat(i, j) = upper_mat(i, j) - mean[j_d];
      }
    }
    if (n_given > 0)
    {
      for (int i = 0; i < n; i++)
      {
        for (int j = 0; j < n_given; j++)
        {
          int j_g = g_to_ind(j);
          given_x_mat(i, j) = given_x_mat(i, j) - mean[j_g];
        }
      }
    }
  }
  
  // Account for marginal distributions if need
  arma::mat q_lower;
  arma::mat q_upper;
  arma::mat q_given;
  arma::mat lower_d_marginal;
  arma::mat upper_d_marginal;
  arma::mat given_d_marginal;
  arma::mat lower_mat0;
  arma::mat upper_mat0;
  arma::mat given_x_mat0;
  arma::field<arma::mat> grad_marginal_list_lower;
  arma::field<arma::mat> grad_marginal_list_upper;
  arma::field<arma::mat> grad_marginal_list_given;
  if (is_marginal)
  {
    // Initialize some matrices
    lower_d_marginal = arma::mat(n, n_dependent);
    upper_d_marginal = arma::mat(n, n_dependent);
    if (n_given > 0)
    {
      given_d_marginal = arma::mat(n, n_given);
      grad_marginal_list_given = arma::field<arma::mat>(n_given);
    }
    grad_marginal_list_lower = arma::field<arma::mat>(n_dependent);
    grad_marginal_list_upper = arma::field<arma::mat>(n_dependent);
    if (grad_sigma)
    {
      q_lower = arma::mat(n, n_dependent);
      q_upper = arma::mat(n, n_dependent);
      lower_mat0 = as<arma::mat>(lower_mat);
      upper_mat0 = as<arma::mat>(upper_mat);
      if (n_given > 0)
      {
        q_given = arma::mat(n, n_given);
        given_x_mat0 = as<arma::mat>(given_x_mat);
      }
    }
    // Calculate adjusted arguments
    for (int i = 0; i < n_dim; i++)
    {
      if (marginal_names[i] != "normal")
      {
        // Preliminary vector to store some
        // derivatives information
        NumericVector grad_tmp_lower;
        NumericVector grad_tmp_upper;
        NumericVector grad_tmp_given;
        arma::mat grad_tmp_marginal_lower;
        arma::mat grad_tmp_marginal_upper;
        arma::mat grad_tmp_marginal_given;
        
        // Impose standardization
        double sigma_sqrt = sqrt(sigma(i, i));
        int i_adj;
        if (given_ind_logical[i])
        {
          i_adj = ind_to_g[i];
          given_x_mat(_, i_adj) = given_x_mat(_, i_adj) / sigma_sqrt;
        }
        else
        {
          i_adj = ind_to_d[i];
          lower_mat(_, i_adj) = lower_mat(_, i_adj) / sigma_sqrt;
          upper_mat(_, i_adj) = upper_mat(_, i_adj) / sigma_sqrt;
        }
        // Logistic distribution
        if (marginal_names[i] == "logistic")
        {
          n_marginal_par[i] = 0;
          double marginal_sd = arma::datum::pi / sqrt(3.0);
          double sd_ratio = marginal_sd / sigma_sqrt;
          if (given_ind_logical[i])
          {
            NumericVector arg_tmp = marginal_sd * given_x_mat(_, i_adj);
            if (is_grad)
            {
              grad_tmp_given = Rcpp::dlogis(arg_tmp, 0.0, 1.0);
              grad_tmp_given = sd_ratio * grad_tmp_given;
            }
            given_x_mat(_, i_adj) = Rcpp::plogis(arg_tmp, 0.0, 1.0);
            given_x_mat(_, i_adj) = Rcpp::qnorm(given_x_mat(_, i_adj), 0.0, 1.0);
            if (grad_sigma)
            {
              NumericVector vec_tmp = given_x_mat(_, i_adj);
              q_given.col(i_adj) = as<arma::vec>(vec_tmp);
            }
          }
          else
          {
            NumericVector arg_tmp_lower = marginal_sd * lower_mat(_, i_adj);
            NumericVector arg_tmp_upper = marginal_sd * upper_mat(_, i_adj);
            if (is_grad)
            {
              grad_tmp_lower = Rcpp::dlogis(arg_tmp_lower, 0.0, 1.0);
              grad_tmp_upper = Rcpp::dlogis(arg_tmp_upper, 0.0, 1.0);
              grad_tmp_lower = sd_ratio * grad_tmp_lower;
              grad_tmp_upper = sd_ratio * grad_tmp_upper;
            }
            lower_mat(_, i_adj) = Rcpp::plogis(arg_tmp_lower, 0.0, 1.0);
            upper_mat(_, i_adj) = Rcpp::plogis(arg_tmp_upper, 0.0, 1.0);
            lower_mat(_, i_adj) = Rcpp::qnorm(lower_mat(_, i_adj), 0.0, 1.0);
            upper_mat(_, i_adj) = Rcpp::qnorm(upper_mat(_, i_adj), 0.0, 1.0);
            if (grad_sigma)
            {
              NumericVector vec_tmp = lower_mat(_, i_adj);
              q_lower.col(i_adj) = as<arma::vec>(vec_tmp);
              vec_tmp = upper_mat(_, i_adj);
              q_upper.col(i_adj) = as<arma::vec>(vec_tmp);
            }
          }
        }
        // Student distribution
        if ((marginal_names[i] == "student") || (marginal_names[i] == "t"))
        {
          double df = marginal_par[i];
          n_marginal_par[i] = 1;
          if (given_ind_logical[i])
          {
            List t_grad = pt0(given_x_mat(_, i_adj), df, 
                              false, true, true, 10);
            if (is_grad)
            {
              grad_tmp_given = t_grad["grad_x"];
              grad_tmp_given = grad_tmp_given / sigma_sqrt;
              NumericMatrix mat_tmp = t_grad["grad_df"];
              grad_tmp_marginal_given = as<arma::mat>(mat_tmp);
            }
            NumericVector t_tmp = t_grad["prob"];
            given_x_mat(_, i_adj) = t_tmp;
            given_x_mat(_, i_adj) = Rcpp::qnorm(given_x_mat(_, i_adj), 0.0, 1.0);
            if (grad_sigma)
            {
              NumericVector vec_tmp = given_x_mat(_, i_adj);
              q_given.col(i_adj) = as<arma::vec>(vec_tmp);
            }
          }
          else
          {
            List t_grad_lower = pt0(lower_mat(_, i_adj), df, 
                                    false, true, true, 10);
            List t_grad_upper = pt0(upper_mat(_, i_adj), df, 
                                    false, true, true, 10);
            if (is_grad)
            {
              grad_tmp_lower = t_grad_lower["grad_x"];
              grad_tmp_lower = grad_tmp_lower / sigma_sqrt;
              grad_tmp_upper = t_grad_upper["grad_x"];
              grad_tmp_upper = grad_tmp_upper / sigma_sqrt;
              NumericMatrix mat_tmp1 = t_grad_lower["grad_df"];
              grad_tmp_marginal_lower = as<arma::mat>(mat_tmp1);
              NumericMatrix mat_tmp2 = t_grad_upper["grad_df"];
              grad_tmp_marginal_upper = as<arma::mat>(mat_tmp2);
            }
            NumericVector t_tmp_lower = t_grad_lower["prob"];
            NumericVector t_tmp_upper = t_grad_upper["prob"];
            lower_mat(_, i_adj) = t_tmp_lower;
            upper_mat(_, i_adj) = t_tmp_upper;
            lower_mat(_, i_adj) = Rcpp::qnorm(lower_mat(_, i_adj), 0.0, 1.0);
            upper_mat(_, i_adj) = Rcpp::qnorm(upper_mat(_, i_adj), 0.0, 1.0);
            if (grad_sigma)
            {
              NumericVector vec_tmp = lower_mat(_, i_adj);
              q_lower.col(i_adj) = as<arma::vec>(vec_tmp);
              vec_tmp = upper_mat(_, i_adj);
              q_upper.col(i_adj) = as<arma::vec>(vec_tmp);
            }
          }
        }
        // PGN distribution
        if ((marginal_names[i] == "PGN") || (marginal_names[i] == "hpa"))
        {
          NumericVector pc = marginal_par[i];
          n_marginal_par[i] = pc.size();
          if (given_ind_logical[i])
          {
            List hpa_grad = hpa::phpa0(given_x_mat(_, i_adj), pc, 
                                       0, 1, false, false, false, true);
            if (is_grad)
            {
              grad_tmp_given = hpa_grad["grad_x"];
              grad_tmp_given = grad_tmp_given / sigma_sqrt;
              NumericMatrix mat_tmp = hpa_grad["grad_pc"];
              grad_tmp_marginal_given = as<arma::mat>(mat_tmp);
            }
            NumericVector hpa_tmp = hpa_grad["prob"];
            given_x_mat(_, i_adj) = hpa_tmp;
            given_x_mat(_, i_adj) = Rcpp::qnorm(given_x_mat(_, i_adj), 0.0, 1.0);
            if (grad_sigma)
            {
              NumericVector vec_tmp = given_x_mat(_, i_adj);
              q_given.col(i_adj) = as<arma::vec>(vec_tmp);
            }
          }
          else
          {
            List hpa_grad_lower = hpa::phpa0(lower_mat(_, i_adj), pc, 
                                             0, 1, false, false, false, true);
            List hpa_grad_upper = hpa::phpa0(upper_mat(_, i_adj), pc, 
                                             0, 1, false, false, false, true);
            if (is_grad)
            {
              grad_tmp_lower = hpa_grad_lower["grad_x"];
              grad_tmp_lower = grad_tmp_lower / sigma_sqrt;
              grad_tmp_upper = hpa_grad_upper["grad_x"];
              grad_tmp_upper = grad_tmp_upper / sigma_sqrt;
              NumericMatrix mat_tmp1 = hpa_grad_lower["grad_pc"];
              grad_tmp_marginal_lower = as<arma::mat>(mat_tmp1);
              NumericMatrix mat_tmp2 = hpa_grad_upper["grad_pc"];
              grad_tmp_marginal_upper = as<arma::mat>(mat_tmp2);
            }
            NumericVector hpa_tmp_lower = hpa_grad_lower["prob"];
            NumericVector hpa_tmp_upper = hpa_grad_upper["prob"];
            lower_mat(_, i_adj) = hpa_tmp_lower;
            upper_mat(_, i_adj) = hpa_tmp_upper;
            lower_mat(_, i_adj) = Rcpp::qnorm(lower_mat(_, i_adj), 0.0, 1.0);
            upper_mat(_, i_adj) = Rcpp::qnorm(upper_mat(_, i_adj), 0.0, 1.0);
            if (grad_sigma)
            {
              NumericVector vec_tmp = lower_mat(_, i_adj);
              q_lower.col(i_adj) = as<arma::vec>(vec_tmp);
              vec_tmp = upper_mat(_, i_adj);
              q_upper.col(i_adj) = as<arma::vec>(vec_tmp);
            }
          }
        }
        // Remove standardization
        if (given_ind_logical[i])
        {
          given_x_mat(_, i_adj) = given_x_mat(_, i_adj) * sigma_sqrt;
          if (is_grad)
          {
            NumericVector den_tmp = Rcpp::dnorm(given_x_mat(_, i_adj), 
                                                0.0, sigma_sqrt);
            grad_tmp_given = grad_tmp_given / den_tmp;
            if (grad_tmp_marginal_given.size() > 0)
            {
              grad_tmp_marginal_given = grad_tmp_marginal_given.each_col() / 
                                        as<arma::vec>(den_tmp);
              grad_marginal_list_given.at(i_adj) = grad_tmp_marginal_given;
              grad_marginal_list_given.at(i_adj).replace(arma::datum::nan, 0);
            }
            given_d_marginal.col(i_adj) = as<arma::vec>(grad_tmp_given);
          }
        }
        else
        {
          lower_mat(_, i_adj) = lower_mat(_, i_adj) * sigma_sqrt;
          upper_mat(_, i_adj) = upper_mat(_, i_adj) * sigma_sqrt;
          if (is_grad)
          {
            NumericVector den_tmp_lower = Rcpp::dnorm(lower_mat(_, i_adj),
                                                      0.0, sigma_sqrt);
            NumericVector den_tmp_upper = Rcpp::dnorm(upper_mat(_, i_adj),
                                                      0.0, sigma_sqrt);
            grad_tmp_lower = grad_tmp_lower / den_tmp_lower;
            grad_tmp_upper = grad_tmp_upper / den_tmp_upper;
            lower_d_marginal.col(i_adj) = as<arma::vec>(grad_tmp_lower);
            upper_d_marginal.col(i_adj) = as<arma::vec>(grad_tmp_upper);
            if (grad_tmp_marginal_lower.size() > 0)
            {
              grad_tmp_marginal_lower = grad_tmp_marginal_lower.each_col() / 
                                        as<arma::vec>(den_tmp_lower);
              grad_tmp_marginal_upper = grad_tmp_marginal_upper.each_col() / 
                                        as<arma::vec>(den_tmp_upper);
              grad_marginal_list_lower.at(i_adj) = grad_tmp_marginal_lower;
              grad_marginal_list_upper.at(i_adj) = grad_tmp_marginal_upper;
              grad_marginal_list_lower.at(i_adj).replace(arma::datum::nan, 0);
              grad_marginal_list_upper.at(i_adj).replace(arma::datum::nan, 0);
            }
          }
        }
      }
    }
    // Substitute some values
    lower_d_marginal.replace(arma::datum::nan, 0);
    upper_d_marginal.replace(arma::datum::nan, 0);
    q_lower.replace(-arma::datum::inf, 0);
    q_upper.replace(arma::datum::inf, 0);
    lower_mat0.replace(-arma::datum::inf, 0);
    upper_mat0.replace(arma::datum::inf, 0);
  }

  // Vector of zero means
  NumericVector mean_zero(n_dim);
  
  // Account for conditioning
  NumericMatrix mean_cond;
  NumericMatrix sigma_cond;
  NumericMatrix lower_g;
  NumericMatrix lower_d;
  NumericMatrix upper_g;
  NumericMatrix upper_d;
  arma::mat s12s22;
  arma::mat diff_mean_by_sigma;
  List cond;
  if (n_given > 0)
  {
    // Get conditional distribution parameters
    List cmnorm_control = List::create(
      Named("diff_mean_by_sigma_dg") = grad_sigma);
    cond = cmnorm(mean_zero, sigma,
                  given_ind, given_x_mat,
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
    lower_d = NumericMatrix(n, n_dependent);
    upper_d = NumericMatrix(n, n_dependent);

    for (int i = 0; i < n_dependent; i++)
    {
      lower_d(_, i) = lower_mat(_, i) - mean_cond(_, i);
      upper_d(_, i) = upper_mat(_, i) - mean_cond(_, i);
    }
  }
  else
  {
    sigma_cond = sigma;
    lower_d = lower_mat;
    upper_d = upper_mat;
  }

  // Transform to arma
  arma::mat const lower_d_arma(lower_d.begin(), n, n_dependent, false);
  arma::mat const upper_d_arma(upper_d.begin(), n, n_dependent, false);
  
  arma::mat const sigma_cond_arma(sigma_cond.begin(), 
                                  n_dependent, n_dependent, false);
  arma::mat sigma_cond_inv;

  // Prepare vector to store probabilities
  arma::vec prob(n);
  
  // Special routine for univariate case
  if (n_dependent == 1)
  {
    double sigma_cond_sqr = sqrt(sigma_cond_arma(0, 0));
    prob = arma::normcdf(upper_d_arma / sigma_cond_sqr) - 
           arma::normcdf(lower_d_arma / sigma_cond_sqr);
  }

  // Special routine for bivariate normal
  // Genz, A. Numerical computation of rectangular bivariate and trivariate 
  // normal and t probabilities. Statistics and Computing 14, 251–260 (2004). 
  // https://doi.org/10.1023/B:STCO.0000035304.20635.31
  // Formula (3) with 30 Gauss-Legendre points.
  if ((n_dependent == 2) & ((method == "Gassmann") || 
                            (method == "default")))
  {
    // Covariance matrix parameters
    double sigma1 = sqrt(sigma_cond_arma.at(0, 0));
    double sigma2 = sqrt(sigma_cond_arma.at(1, 1));
    double cov12 = sigma_cond_arma.at(0, 1);
    double rho = cov12 / (sigma1 * sigma2);

    // Gauss quadrature values
    arma::mat gs = as<arma::mat>(GaussQuadrature(30));
    arma::vec x = (rho * gs.col(0) + rho) / 2;
    arma::vec w = gs.col(1);

    // Standardized integration limits
    arma::vec lwr1 = lower_d_arma.col(0) / sigma1;
    arma::vec lwr2 = lower_d_arma.col(1) / sigma2;
    arma::vec upr1 = upper_d_arma.col(0) / sigma1;
    arma::vec upr2 = upper_d_arma.col(1) / sigma2;
    
    // Preliminary values for vectorization purposes
    arma::vec adj = 1 - pow(x, 2);
    arma::vec adj1 = 1 / (2 * adj);
    arma::vec adj2 = w % (rho / (4 * arma::datum::pi * sqrt(adj)));
    
    // Calculate probability
    arma::vec prob0 = (arma::normcdf(upr1) - arma::normcdf(lwr1)) %
                      (arma::normcdf(upr2) - arma::normcdf(lwr2));

    // Estimation of probabilities for each observation
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_cores) if (n_cores > 1)
    #endif
    for(int i = 0; i < n; i++)
    {
      // Value to aggregate probabilities
      double p = 0;
      
      // Check which elements are infinite
      bool inf_upr1 = std::isinf(upr1.at(i));
      bool inf_upr2 = std::isinf(upr2.at(i));
      bool inf_lwr1 = std::isinf(lwr1.at(i));
      bool inf_lwr2 = std::isinf(lwr2.at(i));
      
      // Phase 1
      if (!(inf_upr1 | inf_upr2))
      {
        double c1 = std::pow(upr1.at(i), 2) + std::pow(upr2.at(i), 2);
        double c2 = 2 * upr1.at(i) * upr2.at(i);
        p = sum(adj2 % exp((c2 * x - c1) % adj1));
      }
      
      // Phase 2
      if (!(inf_lwr1 | inf_upr2))
      {
        double c1 = std::pow(lwr1.at(i), 2) + std::pow(upr2.at(i), 2);
        double c2 = 2 * lwr1.at(i) * upr2.at(i);
        p = p - sum(adj2 % exp((c2 * x - c1) % adj1));
      }
      
      // Phase 3
      if (!(inf_upr1 | inf_lwr2))
      {
        double c1 = std::pow(upr1.at(i), 2) + std::pow(lwr2.at(i), 2);
        double c2 = 2 * upr1.at(i) * lwr2.at(i);
        p = p - sum(adj2 % exp((c2 * x - c1) % adj1));
      }
      
      // Phase 4
      if (!(inf_lwr1 | inf_lwr2))
      {
        double c1 = std::pow(lwr1.at(i), 2) + std::pow(lwr2.at(i), 2);
        double c2 = 2 * lwr1.at(i) * lwr2.at(i);
        p = p + sum(adj2 % exp((c2 * x - c1) % adj1));
      }
      
      prob.at(i) = p;
    }
    
    // Aggregate the results
    prob = prob + prob0;
  }
  
  // Special routine for trivariate normal
  // Genz, A. Numerical computation of rectangular bivariate and trivariate 
  // normal and t probabilities. Statistics and Computing 14, 251–260 (2004). 
  // https://doi.org/10.1023/B:STCO.0000035304.20635.31
  // Formula (14) with 30 Gauss-Legendre points.
  if ((n_dependent == 3) & ((method == "Gassmann") || 
                            (method == "default")))
  {
    // Covariance matrix parameters before permutation
    arma::vec sigma3 = {sqrt(sigma_cond_arma.at(0, 0)),
                        sqrt(sigma_cond_arma.at(1, 1)),
                        sqrt(sigma_cond_arma.at(2, 2))};
    arma::vec cov3 = {sigma_cond_arma.at(0, 1),
                      sigma_cond_arma.at(0, 2),
                      sigma_cond_arma.at(1, 2)};
    arma::vec cor3 = {cov3.at(0) / (sigma3.at(0) * sigma3.at(1)),
                      cov3.at(1) / (sigma3.at(0) * sigma3.at(2)),
                      cov3.at(2) / (sigma3.at(1) * sigma3.at(2))};
    arma::mat cor3_mat = {{1,          cor3.at(0), cor3.at(1)},
                          {cor3.at(0), 1,          cor3.at(2)},
                          {cor3.at(1), cor3.at(2), 1}};
    
    // Get optimal permutation
    arma::vec cor_tmp = {std::max(std::abs(cor3.at(0)), std::abs(cor3.at(1))),
                         std::max(std::abs(cor3.at(0)), std::abs(cor3.at(2))),
                         std::max(std::abs(cor3.at(1)), std::abs(cor3.at(2)))};
    arma::uvec ind_sorted = sort_index(cor_tmp, "ascend");
    
    // Assign values accounting for permutations
    sigma3 = sigma3.elem(ind_sorted);
    cov3 = cov3.elem(ind_sorted);
    cor3.at(0) = cor3_mat.at(ind_sorted.at(0), ind_sorted.at(1));
    cor3.at(1) = cor3_mat.at(ind_sorted.at(0), ind_sorted.at(2));
    cor3.at(2) = cor3_mat.at(ind_sorted.at(1), ind_sorted.at(2));
    arma::vec cor3_sqr = arma::pow(cor3, 2);
    arma::mat lwr = lower_d_arma.cols(ind_sorted);
    arma::mat upr = upper_d_arma.cols(ind_sorted);
    for (int i = 0; i <= 2; i++)
    {
      lwr.col(i) = lwr.col(i) / sigma3.at(i);
      upr.col(i) = upr.col(i) / sigma3.at(i);
    }

    // Calculate first part of the probability
    NumericMatrix lwr_biv = wrap(lwr.cols(1, 2));
    NumericMatrix upr_biv = wrap(upr.cols(1, 2));
    NumericMatrix sigma_biv(2, 2);
    sigma_biv(0, 0) = 1;
    sigma_biv(0, 1) = cor3.at(2);
    sigma_biv(1, 0) = cor3.at(2);
    sigma_biv(1, 1) = 1;
    NumericVector mean_biv = NumericVector::create(0, 0);
    List prob0_list = pmnorm(lwr_biv, upr_biv, 
                             NumericVector(),
                             mean_biv, sigma_biv,
                             NumericVector(), n_sim,
                             "default", "mean",
                             false, false, false, false, false, false,
                             R_NilValue, n_cores);
    arma::vec prob0 = prob0_list["prob"];
    prob0 = prob0 % (arma::normcdf(upr.col(0)) - arma::normcdf(lwr.col(0)));

    // Gauss quadrature values accounting for change
    // of variables from [-1, 1] to [0, 1]
    int n_gs = 30;
    arma::mat gs = as<arma::mat>(GaussQuadrature(n_gs));
    arma::vec x = (gs.col(0) + 1) / 2;
    arma::vec w = gs.col(1);
    
    // Preliminary values for vectorization purposes
    arma::vec x_sqr = arma::pow(x, 2);
    arma::vec two_cor12_x = 2 * cor3.at(0) * x;
    arma::vec two_cor13_x = 2 * cor3.at(1) * x;
    arma::vec cor12_x_sqr = cor3_sqr.at(0) * x_sqr;
    arma::vec cor13_x_sqr = cor3_sqr.at(1) * x_sqr;
    arma::vec one_minus_cor12_x_sqr = 1 - cor12_x_sqr;
    arma::vec one_minus_cor13_x_sqr = 1 - cor13_x_sqr;
    arma::vec two_one_minus_cor12_x_sqr = 2 * one_minus_cor12_x_sqr;
    arma::vec two_one_minus_cor13_x_sqr = 2 * one_minus_cor13_x_sqr;
    arma::vec two_x_sqr_cor = 2 * cor3.at(0) * cor3.at(1) * cor3.at(2) * x_sqr;
    arma::vec u_denom_share = one_minus_cor12_x_sqr - cor13_x_sqr - 
                              cor3_sqr.at(2) + two_x_sqr_cor;
    arma::vec u2_denom = arma::sqrt(one_minus_cor13_x_sqr % u_denom_share);
    arma::vec u3_denom = arma::sqrt(one_minus_cor12_x_sqr % u_denom_share);
    arma::vec u_nom_share = cor3.at(0) * cor3.at(1) * x_sqr - cor3.at(2);
    arma::vec u2_nom_sp = (cor3.at(1) * cor3.at(2) - cor3.at(0)) * x;
    arma::vec u3_nom_sp = (cor3.at(0) * cor3.at(2) - cor3.at(1)) * x;
    // divide by 4 instead of 2 because of change of variable in the quadrature
    arma::vec w2 = w % ((cor3.at(1) / (4 * arma::datum::pi)) / 
                   arma::sqrt(one_minus_cor13_x_sqr));
    arma::vec w3 = w % ((cor3.at(0) / (4 * arma::datum::pi)) / 
                   arma::sqrt(one_minus_cor12_x_sqr));
    
    // Some functions of integration limits
    arma::mat lwr_sqr = arma::pow(lwr, 2);
    arma::mat upr_sqr = arma::pow(upr, 2);
    
    // Start calculations
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_cores) if (n_cores > 1)
    #endif
    for(int i = 0; i < n; i++)
    {
      // Initialize values to store probabilities
      double p2 = 0;
      double p3 = 0;
      
      // Check which elements are infinite
      bool inf_upr_1 = std::isinf(upr.at(i, 0));
      bool inf_upr_2 = std::isinf(upr.at(i, 1));
      bool inf_upr_3 = std::isinf(upr.at(i, 2));
      bool inf_lwr_1 = std::isinf(lwr.at(i, 0));
      bool inf_lwr_2 = std::isinf(lwr.at(i, 1));
      bool inf_lwr_3 = std::isinf(lwr.at(i, 2));
      
      //
      // Phase 1
      //
      
      // First component                          
      if (!(inf_upr_1 | inf_upr_3))
      {
        arma::vec p_u2_upr(n_gs, arma::fill::ones);
        arma::vec p_u2_lwr(n_gs, arma::fill::zeros);
        if (!inf_upr_2)
        {
          arma::vec u2_nom = upr.at(i, 1) * one_minus_cor13_x_sqr + 
                             upr.at(i, 0) * u2_nom_sp +
                             upr.at(i, 2) * u_nom_share;
          arma::vec u2 = u2_nom / u2_denom;
          p_u2_upr = arma::normcdf(u2);
        }
        if (!inf_lwr_2)
        {
          arma::vec u2_nom = lwr.at(i, 1) * one_minus_cor13_x_sqr + 
                             upr.at(i, 0) * u2_nom_sp +
                             upr.at(i, 2) * u_nom_share;
          arma::vec u2 = u2_nom / u2_denom;
          p_u2_lwr = arma::normcdf(u2);
        }
        arma::vec p_u2 = p_u2_upr - p_u2_lwr;
        arma::vec f2 = arma::exp((two_cor13_x * (upr.at(i, 0) * upr.at(i, 2)) - 
                                  (upr_sqr.at(i, 0) + upr_sqr.at(i, 2))) / 
                                 two_one_minus_cor13_x_sqr);
        p2 = sum(w2 % f2 % p_u2);
      }
      
      // Second component
      if (!(inf_upr_1 | inf_upr_2))
      {
        arma::vec p_u3_upr(n_gs, arma::fill::ones);
        arma::vec p_u3_lwr(n_gs, arma::fill::zeros);
        if (!inf_upr_3)
        {
          arma::vec u3_nom = upr.at(i, 2) * one_minus_cor12_x_sqr + 
                             upr.at(i, 0) * u3_nom_sp +
                             upr.at(i, 1) * u_nom_share;
          arma::vec u3 = u3_nom / u3_denom;
          p_u3_upr = arma::normcdf(u3);
        }
        if (!inf_lwr_3)
        {
          arma::vec u3_nom = lwr.at(i, 2) * one_minus_cor12_x_sqr + 
                             upr.at(i, 0) * u3_nom_sp +
                             upr.at(i, 1) * u_nom_share;
          arma::vec u3 = u3_nom / u3_denom;
          p_u3_lwr = arma::normcdf(u3);
        }
        arma::vec p_u3 = p_u3_upr - p_u3_lwr;
        arma::vec f3 = arma::exp((two_cor12_x * (upr.at(i, 0) * upr.at(i, 1)) - 
                                  (upr_sqr.at(i, 0) + upr_sqr.at(i, 1))) / 
                                 two_one_minus_cor12_x_sqr);
        p3 = sum(w3 % f3 % p_u3);
      }
      
      //
      // Phase 2
      //
      
      // First component                          
      if (!(inf_lwr_1 | inf_upr_3))
      {
        arma::vec p_u2_upr(n_gs, arma::fill::ones);
        arma::vec p_u2_lwr(n_gs, arma::fill::zeros);
        if (!inf_upr_2)
        {
          arma::vec u2_nom = upr.at(i, 1) * one_minus_cor13_x_sqr + 
                             lwr.at(i, 0) * u2_nom_sp +
                             upr.at(i, 2) * u_nom_share;
          arma::vec u2 = u2_nom / u2_denom;
          p_u2_upr = arma::normcdf(u2);
        }
        if (!inf_lwr_2)
        {
          arma::vec u2_nom = lwr.at(i, 1) * one_minus_cor13_x_sqr + 
                             lwr.at(i, 0) * u2_nom_sp +
                             upr.at(i, 2) * u_nom_share;
          arma::vec u2 = u2_nom / u2_denom;
          p_u2_lwr = arma::normcdf(u2);
        }
        arma::vec p_u2 = p_u2_upr - p_u2_lwr;
        arma::vec f2 = arma::exp((two_cor13_x * (lwr.at(i, 0) * upr.at(i, 2)) - 
                                  (lwr_sqr.at(i, 0) + upr_sqr.at(i, 2))) / 
                                 two_one_minus_cor13_x_sqr);
        p2 = p2 - sum(w2 % f2 % p_u2);
      }
      
      // Second component
      if (!(inf_lwr_1 | inf_upr_2))
      {
        arma::vec p_u3_upr(n_gs, arma::fill::ones);
        arma::vec p_u3_lwr(n_gs, arma::fill::zeros);
        if (!inf_upr_3)
        {
          arma::vec u3_nom = upr.at(i, 2) * one_minus_cor12_x_sqr + 
                             lwr.at(i, 0) * u3_nom_sp +
                             upr.at(i, 1) * u_nom_share;
          arma::vec u3 = u3_nom / u3_denom;
          p_u3_upr = arma::normcdf(u3);
        }
        if (!inf_lwr_3)
        {
          arma::vec u3_nom = lwr.at(i, 2) * one_minus_cor12_x_sqr + 
                             lwr.at(i, 0) * u3_nom_sp +
                             upr.at(i, 1) * u_nom_share;
          arma::vec u3 = u3_nom / u3_denom;
          p_u3_lwr = arma::normcdf(u3);
        }
        arma::vec p_u3 = p_u3_upr - p_u3_lwr;
        arma::vec f3 = arma::exp((two_cor12_x * (lwr.at(i, 0) * upr.at(i, 1)) - 
                                  (lwr_sqr.at(i, 0) + upr_sqr.at(i, 1))) / 
                                  two_one_minus_cor12_x_sqr);
        p3 = p3 - sum(w3 % f3 % p_u3);
      }
      
      //
      // Phase 3
      //
      
      // First component                          
      if (!(inf_upr_1 | inf_lwr_3))
      {
        arma::vec p_u2_upr(n_gs, arma::fill::ones);
        arma::vec p_u2_lwr(n_gs, arma::fill::zeros);
        if (!inf_upr_2)
        {
          arma::vec u2_nom = upr.at(i, 1) * one_minus_cor13_x_sqr + 
                             upr.at(i, 0) * u2_nom_sp +
                             lwr.at(i, 2) * u_nom_share;
          arma::vec u2 = u2_nom / u2_denom;
          p_u2_upr = arma::normcdf(u2);
        }
        if (!inf_lwr_2)
        {
          arma::vec u2_nom = lwr.at(i, 1) * one_minus_cor13_x_sqr + 
                             upr.at(i, 0) * u2_nom_sp +
                             lwr.at(i, 2) * u_nom_share;
          arma::vec u2 = u2_nom / u2_denom;
          p_u2_lwr = arma::normcdf(u2);
        }
        arma::vec p_u2 = p_u2_upr - p_u2_lwr;
        arma::vec f2 = arma::exp((two_cor13_x * (upr.at(i, 0) * lwr.at(i, 2)) - 
                                  (upr_sqr.at(i, 0) + lwr_sqr.at(i, 2))) / 
                                  two_one_minus_cor13_x_sqr);
        p2 = p2 - sum(w2 % f2 % p_u2);
      }
      
      // Second component
      if (!(inf_upr_1 | inf_lwr_2))
      {
        arma::vec p_u3_upr(n_gs, arma::fill::ones);
        arma::vec p_u3_lwr(n_gs, arma::fill::zeros);
        if (!inf_upr_3)
        {
          arma::vec u3_nom = upr.at(i, 2) * one_minus_cor12_x_sqr + 
                             upr.at(i, 0) * u3_nom_sp +
                             lwr.at(i, 1) * u_nom_share;
          arma::vec u3 = u3_nom / u3_denom;
          p_u3_upr = arma::normcdf(u3);
        }
        if (!inf_lwr_3)
        {
          arma::vec u3_nom = lwr.at(i, 2) * one_minus_cor12_x_sqr + 
                             upr.at(i, 0) * u3_nom_sp +
                             lwr.at(i, 1) * u_nom_share;
          arma::vec u3 = u3_nom / u3_denom;
          p_u3_lwr = arma::normcdf(u3);
        }
        arma::vec p_u3 = p_u3_upr - p_u3_lwr;
        arma::vec f3 = arma::exp((two_cor12_x * (upr.at(i, 0) * lwr.at(i, 1)) - 
                                (upr_sqr.at(i, 0) + lwr_sqr.at(i, 1))) / 
                                two_one_minus_cor12_x_sqr);
        p3 = p3 - sum(w3 % f3 % p_u3);
      }
      
      //
      // Phase 4
      //
      
      // First component                          
      if (!(inf_lwr_1 | inf_lwr_3))
      {
        arma::vec p_u2_upr(n_gs, arma::fill::ones);
        arma::vec p_u2_lwr(n_gs, arma::fill::zeros);
        if (!inf_upr_2)
        {
          arma::vec u2_nom = upr.at(i, 1) * one_minus_cor13_x_sqr + 
                             lwr.at(i, 0) * u2_nom_sp +
                             lwr.at(i, 2) * u_nom_share;
          arma::vec u2 = u2_nom / u2_denom;
          p_u2_upr = arma::normcdf(u2);
        }
        if (!inf_lwr_2)
        {
          arma::vec u2_nom = lwr.at(i, 1) * one_minus_cor13_x_sqr + 
                             lwr.at(i, 0) * u2_nom_sp +
                             lwr.at(i, 2) * u_nom_share;
          arma::vec u2 = u2_nom / u2_denom;
          p_u2_lwr = arma::normcdf(u2);
        }
        arma::vec p_u2 = p_u2_upr - p_u2_lwr;
        arma::vec f2 = arma::exp((two_cor13_x * (lwr.at(i, 0) * lwr.at(i, 2)) - 
                                 (lwr_sqr.at(i, 0) + lwr_sqr.at(i, 2))) / 
                                 two_one_minus_cor13_x_sqr);
        p2 = p2 + sum(w2 % f2 % p_u2);
      }
      
      // Second component
      if (!(inf_lwr_1 | inf_lwr_2))
      {
        arma::vec p_u3_upr(n_gs, arma::fill::ones);
        arma::vec p_u3_lwr(n_gs, arma::fill::zeros);
        if (!inf_upr_3)
        {
          arma::vec u3_nom = upr.at(i, 2) * one_minus_cor12_x_sqr + 
                             lwr.at(i, 0) * u3_nom_sp +
                             lwr.at(i, 1) * u_nom_share;
          arma::vec u3 = u3_nom / u3_denom;
          p_u3_upr = arma::normcdf(u3);
        }
        if (!inf_lwr_3)
        {
          arma::vec u3_nom = lwr.at(i, 2) * one_minus_cor12_x_sqr + 
                             lwr.at(i, 0) * u3_nom_sp +
                             lwr.at(i, 1) * u_nom_share;
          arma::vec u3 = u3_nom / u3_denom;
          p_u3_lwr = arma::normcdf(u3);
        }
        arma::vec p_u3 = p_u3_upr - p_u3_lwr;
        arma::vec f3 = arma::exp((two_cor12_x * (lwr.at(i, 0) * lwr.at(i, 1)) - 
                                  (lwr_sqr.at(i, 0) + lwr_sqr.at(i, 1))) / 
                                  two_one_minus_cor12_x_sqr);
        p3 = p3 + sum(w3 % f3 % p_u3);
      }
      
      // Aggregate the results
      prob.at(i) = p2 + p3;
    }
    
    // Sum probabilities
    prob = prob + prob0;
  }
  
  // Special routine for 4-5 dimensional probabilities
  // Gassmann (2003), matrix 5 representation (1-split)
  // if ((((n_dependent == 4) || (n_dependent == 5)) & (method == "default")) ||
  //     ((method == "Gassmann") & (n_dependent > 3)))
  if ((method == "Gassmann") & (n_dependent > 3))
  {
    // Method
    String method_gassmann = "Gassmann";
    
    // Standardize integration limits
    arma::mat lwr = arma::mat(n, n_dependent);
    arma::mat upr = arma::mat(n, n_dependent);
    arma::vec sd_val = arma::sqrt(arma::diagvec(sigma_cond_arma));
    for (int i = 0; i < n_dependent; i++)
    {
      lwr.col(i) = lower_d_arma.col(i) / sd_val.at(i);
      upr.col(i) = upper_d_arma.col(i) / sd_val.at(i);
    }
    
    // Convert covariance matrix to correlation matrix
    arma::mat cor = arma::mat(n_dependent, n_dependent);
    for (int i = 0; i < n_dependent; i++)
    {
      for (int j = 0; j <= i; j++)
      {
        cor.at(i, j) = sigma_cond_arma.at(i, j) / (sd_val.at(i) * sd_val.at(j));
        cor.at(j, i) = cor.at(i, j);
      }
    }
    
    // Calculate first part of the probability
    NumericMatrix lwr_P = wrap(lwr.submat(0, 0, n - 1, n_dependent - 2));
    NumericMatrix upr_P = wrap(upr.submat(0, 0, n - 1, n_dependent - 2));
    arma::mat cor_P_arma = cor.submat(0, 0, n_dependent - 2, n_dependent - 2);
    NumericMatrix cor_P = wrap(cor_P_arma);
    NumericVector mean_P = NumericVector(n_dependent - 1);
    if (n_dependent <= 4)
    {
      method_gassmann = "default";
    }
    List prob0_list = pmnorm(lwr_P, upr_P, 
                             NumericVector(),
                             mean_P, cor_P,
                             NumericVector(), n_sim,
                             method_gassmann, "mean",
                             false, false, false, false, false, false,
                             R_NilValue, n_cores);
    if (n_dependent <= 5)
    {
      method_gassmann = "default";
    }
    arma::vec prob0 = prob0_list["prob"];
    prob0 = prob0 % (arma::normcdf(upr.col(n_dependent - 1)) - 
                     arma::normcdf(lwr.col(n_dependent - 1)));
    
    // Insert a row of zero correlations to cor_P
    cor_P_arma = cor * 1.0;
    for (int i = 0; i < (n_dependent - 1); i++)
    {
      cor_P_arma.at(n_dependent - 1, i) = 0;
      cor_P_arma.at(i, n_dependent - 1) = 0;
    }
    
    // Gauss quadrature values accounting for change
    // of variables from [-1, 1] to [0, 1]
    int n_gs = 30;
    arma::mat gs = as<arma::mat>(GaussQuadrature(n_gs));
    arma::vec x = (gs.col(0) + 1) / 2;
    arma::vec w = gs.col(1) / 2;

    // Calculate the second part of the probability
    arma::vec prob1(n);
    NumericVector mean_zero_2 = NumericVector(2);
    NumericVector mean_zero_d2 = NumericVector(n_dependent);
    arma::vec val(n);
    for(int j = 0; j < (n_dependent - 1); j++)
    {
      // indexes
      arma::uvec ind = {(unsigned int)j, (unsigned int)(n_dependent - 1)};
      arma::uvec noind(n_dependent - 2);
      int counter = 0;
      for (int u1 = 0; u1 < (n_dependent - 1); u1++)
      {
        if (u1 != j)
        {
          noind.at(counter) = (unsigned int)u1;
          counter++;
        }
      }
      NumericVector ind_vec = wrap(ind + 1);
      // submatrices
        // non-changing
      arma::mat upr_noind_arma  = upr.cols(noind);
      NumericMatrix upr_noind_mat = wrap(upr_noind_arma);
      arma::mat lwr_noind_arma  = lwr.cols(noind);
      NumericMatrix lwr_noind_mat = wrap(lwr_noind_arma);
        // for phases
      arma::mat upr_ind_arma = upr.cols(ind);
      arma::mat lwr_ind_arma = lwr.cols(ind);
      NumericMatrix uu_ind_mat = wrap(upr_ind_arma);
      NumericMatrix ll_ind_mat = wrap(lwr_ind_arma);
      arma::mat ul_ind_arma(n, 2);
      ul_ind_arma.col(0) = upr.col(ind.at(0));
      ul_ind_arma.col(1) = lwr.col(ind.at(1));
      NumericMatrix ul_ind_mat = wrap(ul_ind_arma);
      arma::mat lu_ind_arma(n, 2);
      lu_ind_arma.col(0) = lwr.col(ind.at(0));
      lu_ind_arma.col(1) = upr.col(ind.at(1));
      NumericMatrix lu_ind_mat = wrap(lu_ind_arma);
        // select observations
      LogicalVector is_use_uu = (uu_ind_mat(_, 0) != R_PosInf) &
                                (uu_ind_mat(_, 1) != R_PosInf);
      int is_use_uu_n = sum(is_use_uu);
      List control_uu = List::create(Named("is_use") = is_use_uu);
          //
      LogicalVector is_use_ul = (ul_ind_mat(_, 0) != R_PosInf) &
                                (ul_ind_mat(_, 1) != R_NegInf);
      List control_ul = List::create(Named("is_use") = is_use_ul);
      int is_use_ul_n = sum(is_use_ul);
          //
      LogicalVector is_use_lu = (lu_ind_mat(_, 0) != R_NegInf) &
                                (lu_ind_mat(_, 1) != R_PosInf);
      List control_lu = List::create(Named("is_use") = is_use_lu);
      int is_use_lu_n = sum(is_use_lu);
          //
      LogicalVector is_use_ll = (ll_ind_mat(_, 0) != R_NegInf) &
                                (ll_ind_mat(_, 1) != R_NegInf);
      List control_ll = List::create(Named("is_use") = is_use_ll);
      int is_use_ll_n = sum(is_use_ll);
      // Main integration routine
      for (int t = 0; t < n_gs; t++)
      {
        // aggregation variable
        arma::vec prob_j(n);
        
        // omega
        arma::mat omega_arma = (1 - x.at(t)) * cor_P_arma + x.at(t) * cor;
        NumericMatrix omega = wrap(omega_arma);
        arma::mat omega_ind_arma = omega_arma.submat(ind, ind);
        NumericMatrix omega_ind = wrap(omega_ind_arma);
        
        // -------
        // Phase 1
        // -------
        
        if (is_use_uu_n != 0)
        {
          // density
          List p11_list = dmnorm(uu_ind_mat,
                                 mean_zero_2, omega_ind,
                                 NumericVector(),
                                 false, false, false, false,
                                 control_uu, n_cores);
          arma::vec p11 = p11_list["den"];
          // probability
          List p12_list = pmnorm(lwr_noind_mat, upr_noind_mat,
                                 uu_ind_mat,
                                 mean_zero_d2, omega,
                                 ind_vec, n_sim,
                                 method_gassmann, "mean",
                                 false, false, false, false, false, false,
                                 control_uu, n_cores);
          arma::vec p12 = p12_list["prob"];
          // aggregation
          prob_j = prob_j + p11 % p12;
        }
        
        // -------
        // Phase 2
        // -------

        if (is_use_ul_n != 0)
        {
          // density
          List p21_list = dmnorm(ul_ind_mat,
                                 mean_zero_2, omega_ind,
                                 NumericVector(),
                                 false, false, false, false,
                                 control_ul, n_cores);
          arma::vec p21 = p21_list["den"];
          // probability
          List p22_list = pmnorm(lwr_noind_mat, upr_noind_mat,
                                 ul_ind_mat,
                                 mean_zero_d2, omega,
                                 ind_vec, n_sim,
                                 method_gassmann, "mean",
                                 false, false, false, false, false, false,
                                 control_ul, n_cores);
          arma::vec p22 = p22_list["prob"];
          // aggregation
          prob_j = prob_j - p21 % p22;
        }

        // -------
        // Phase 3
        // -------

        if (is_use_lu_n != 0)
        {
          // density
          List p31_list = dmnorm(lu_ind_mat,
                                 mean_zero_2, omega_ind,
                                 NumericVector(),
                                 false, false, false, false,
                                 control_lu, n_cores);
          arma::vec p31 = p31_list["den"];
          // probability
          List p32_list = pmnorm(lwr_noind_mat, upr_noind_mat,
                                 lu_ind_mat,
                                 mean_zero_d2, omega,
                                 ind_vec, n_sim,
                                 method_gassmann, "mean",
                                 false, false, false, false, false, false,
                                 control_lu, n_cores);
          arma::vec p32 = p32_list["prob"];
          // aggregation
          prob_j = prob_j - p31 % p32;
        }

        // -------
        // Phase 4
        // -------

        if (is_use_ll_n != 0)
        {
          // density
          List p41_list = dmnorm(ll_ind_mat,
                                 mean_zero_2, omega_ind,
                                 NumericVector(),
                                 false, false, false, false,
                                 control_ll, n_cores);
          arma::vec p41 = p41_list["den"];
          // probability
          List p42_list = pmnorm(lwr_noind_mat, upr_noind_mat,
                                 ll_ind_mat,
                                 mean_zero_d2, omega,
                                 ind_vec, n_sim,
                                 method_gassmann, "mean",
                                 false, false, false, false, false, false,
                                 control_ll, n_cores);
          arma::vec p42 = p42_list["prob"];
          // aggregation
          prob_j = prob_j + p41 % p42;
        }
        
        // Aggregation
        prob1 = prob1 + prob_j * (cor.at(j, n_dependent - 1) * w.at(t));
      }
    }
      // Sum probabilities
      prob = prob0 + prob1;
  }
  
  // Special routine for multivariate case (at least 5 dimensions)
  if (((method == "default") & (n_dependent > 3)) || 
      ((method == "GHK") & (n_dependent > 1)))
  {
    NumericMatrix random_sequence;
    bool is_random_sequence = control1.containsElementNamed("random_sequence");
    // Generate Halton sequence
    if (!is_random_sequence)
    {
      random_sequence = halton(n_sim, seqPrimes(n_dependent), 100,
                               "NO", "richtmyer", "NO", 
                               false, n_cores);
    }
    else
    {
      NumericMatrix random_sequence_tmp = control1["random_sequence"];
      random_sequence = random_sequence_tmp;
    }
    arma::mat h(as<arma::mat>(random_sequence));
    if (!is_random_sequence)
    {
      h.reshape(random_sequence.size() / n_dependent, n_dependent);
    }
    else
    {
      h.reshape(n_sim, n_dependent);
    }

    // Estimate the probabilities
    // for each observation
    for(int i = 0; i < n; i++)
    {
      prob.at(i) = GHK(lower_d(i, _), upper_d(i, _), sigma_cond, h, 
                       ordering, n_sim, n_cores);
      
    }
  }

  // Account for logarithm for probabilities
  // and save them into the output matrix
  arma::vec prob_log;
  if (log)
  {
    prob_log = arma::log(prob);
    return_list["prob"] = prob_log;
  }
  else
  {
    return_list["prob"] = prob;
  }

  // -------------------------
  // Gradient related stuff
  // -------------------------
  
  // If no gradients are need then return the results
  if (!is_grad)
  {
    return_list.attr("class") = "mnorm_pmnorm";
    return(return_list);
  }

  // Special control variables
  List control_special;
  
  // Create vector of zero means
  NumericVector mean_zero_d = NumericVector(n_dependent);
  
  // Vector of conditional standard deviations
  arma::vec sd_cond_arma = arma::sqrt(arma::diagvec(sigma_cond_arma));
  
  // Convert matrix of conditioned values to arma
  arma::mat const given_x_arma(given_x_mat.begin(), n, n_given, false);
  
  // Matrix to store gradient for variances
  arma::mat grad_var(n, n_dim);
  arma::mat grad_var_marginal;
  if (is_marginal)
  {
    grad_var_marginal = arma::mat(n, n_dim);
  }

  // Variable to control for observations need to calculate
  // appropriate parts of Jacobian
  LogicalVector is_use1;
  int is_use_n1;

  // Estimate gradient respect to lower integration limits
  arma::mat grad_lower_arma(n, n_dependent);
  arma::mat grad_lower_arma0;
  if (is_marginal)
  {
    grad_lower_arma0 = arma::mat(n, n_dependent);
  }
  List cdf_lower_list;
  for (int i = 0; i < n_dependent; i++)
  {
    arma::vec pdf_lower = -arma::normpdf(lower_d_arma.col(i) /
                                         sd_cond_arma.at(i)) /
                           sd_cond_arma.at(i);
    if (n_dependent == 1)
    {
      grad_lower_arma.col(i) = pdf_lower;
    }
    else
    {
      // Select observations for which
      // this contribution is not zero
      is_use1 = (lower_d(_, i) != R_NegInf);
      is_use_n1 = sum(is_use1);
      control_special["is_use"] = is_use1;
      control_special["is_use_n"] = is_use_n1;
      // Prepare data
      NumericMatrix lower_d_new(n, n_dependent - 1); 
      NumericMatrix upper_d_new(n, n_dependent - 1);
      int counter = 0;
      for (int j = 0; j < n_dependent; j++)
      {
        if (j != i)
        {
          lower_d_new(_, counter) = lower_d(_, j);
          upper_d_new(_, counter) = upper_d(_, j);
          counter++;
        }
      }
      NumericVector given_x_new = lower_d(_, i);
      // Estimate the contribution for these observations
      cdf_lower_list = pmnorm(lower_d_new, upper_d_new,
                              given_x_new,
                              mean_zero_d, sigma_cond, 
                              NumericVector::create(i + 1),
                              n_sim, method, ordering, false,
                              false, false, false, false,
                              false,
                              control_special, n_cores);
      arma::vec cdf_lower = cdf_lower_list["prob"];
      grad_lower_arma.col(i) = pdf_lower % cdf_lower;
    }
    if (is_marginal)
    {
      grad_lower_arma0.col(i) = grad_lower_arma.col(i);
      if (marginal_names[d_to_ind(i)] != "normal")
      {
        grad_lower_arma.col(i) = grad_lower_arma.col(i) % 
                                 lower_d_marginal.col(i);
      }
    }
    // Contribute to variance derivative
    if (grad_sigma)
    {
      int i_d = d_to_ind.at(i);
      arma::vec lwr_adj_tmp;
      lwr_adj_tmp = -lower_d_arma.col(i) / (2 * sigma_cond_arma.at(i, i));
      lwr_adj_tmp.replace(arma::datum::inf, 0);
      if (is_marginal)
      {
        grad_var.col(i_d) = grad_var.col(i_d) + 
                            grad_lower_arma0.col(i) % lwr_adj_tmp;
        if (marginal_names[d_to_ind(i)] != "normal")
        {
          arma::vec grad_var_marginal_tmp = 
            (grad_lower_arma0.col(i) % q_lower.col(i)) / 
            (2 * sqrt(sigma(i_d, i_d))) -
            (grad_lower_arma.col(i) % lower_mat0.col(i)) / 
            (2 * sigma(i_d, i_d));
          grad_var_marginal.col(i_d) = grad_var_marginal.col(i_d) +
                                       grad_var_marginal_tmp;
          grad_var.col(i_d) = grad_var.col(i_d) + grad_var_marginal_tmp;
        }
      }
      else
      {
        grad_var.col(i_d) = grad_var.col(i_d) + 
                            grad_lower_arma.col(i) % lwr_adj_tmp;
      }
    }
  }

  // Estimate gradient respect to upper integration limits
  arma::mat grad_upper_arma(n, n_dependent);
  arma::mat grad_upper_arma0;
  if (is_marginal)
  {
    grad_upper_arma0 = arma::mat(n, n_dependent);
  }
  List cdf_upper_list;
  for (int i = 0; i < n_dependent; i++)
  {
    arma::vec pdf_upper = arma::normpdf(upper_d_arma.col(i) /
                                        sd_cond_arma.at(i)) /
                          sd_cond_arma.at(i);      
    if (n_dependent == 1)
    {
      grad_upper_arma.col(i) = pdf_upper;
    }
    else
    {
      // Select observations for which
      // this contribution is not zero
      is_use1 = (upper_d(_, i) != R_PosInf);
      is_use_n1 = sum(is_use1);
      control_special["is_use"] = is_use1;
      control_special["is_use_n"] = is_use_n1;
      // Prepare data
      NumericMatrix lower_d_new(n, n_dependent - 1); 
      NumericMatrix upper_d_new(n, n_dependent - 1);
      int counter = 0;
      for (int j = 0; j < n_dependent; j++)
      {
        if (j != i)
        {
          lower_d_new(_, counter) = lower_d(_, j);
          upper_d_new(_, counter) = upper_d(_, j);
          counter++;
        }
      }
      NumericVector given_x_new = upper_d(_, i);
      // Estimate the contribution for these observations
      cdf_upper_list = pmnorm(lower_d_new, upper_d_new, 
                              given_x_new,
                              mean_zero_d, sigma_cond, 
                              NumericVector::create(i + 1),
                              n_sim, method, ordering, false,
                              false, false, false, false,
                              false, 
                              control_special, n_cores);
      arma::vec cdf_upper = cdf_upper_list["prob"];
      grad_upper_arma.col(i) = pdf_upper % cdf_upper;
    }
    if (is_marginal)
    {
      grad_upper_arma0.col(i) = grad_upper_arma.col(i);
      if (marginal_names[d_to_ind(i)] != "normal")
      {
        grad_upper_arma.col(i) = grad_upper_arma.col(i) % 
                                 upper_d_marginal.col(i);
      }
    }
    // Contribute to variance derivative
    if (grad_sigma)
    {
      int i_d = d_to_ind.at(i);
      arma::vec upr_adj_tmp;
      upr_adj_tmp = -upper_d_arma.col(i) / (2 * sigma_cond_arma.at(i, i));
      upr_adj_tmp.replace(-arma::datum::inf, 0);
      if (is_marginal)
      {
        grad_var.col(i_d) = grad_var.col(i_d) + 
                            grad_upper_arma0.col(i) % upr_adj_tmp;
        if (marginal_names[d_to_ind(i)] != "normal")
        {
          arma::vec grad_var_marginal_tmp = 
            (grad_upper_arma0.col(i) % q_upper.col(i)) / 
            (2 * sqrt(sigma(i_d, i_d))) -
            (grad_upper_arma.col(i) % upper_mat0.col(i)) / 
            (2 * sigma(i_d, i_d));
          grad_var_marginal.col(i_d) = grad_var_marginal.col(i_d) +
                                       grad_var_marginal_tmp;
          grad_var.col(i_d) = grad_var.col(i_d) + grad_var_marginal_tmp;
        }
      }
      else
      {
        grad_var.col(i_d) = grad_var.col(i_d) + 
                            grad_upper_arma.col(i) % upr_adj_tmp;
      }
    }
  }

  // Estimate the gradient respect to conditional values
  arma::mat grad_given_arma;
  arma::mat grad_given_arma0;
  arma::mat grad_sum;
  if ((grad_given || grad_sigma || grad_marginal) & (n_given > 0))
  {
    if (is_marginal)
    {
      grad_sum = grad_upper_arma0 + grad_lower_arma0;
      grad_given_arma = grad_sum * (-s12s22);
      if (grad_marginal)
      {
        grad_given_arma0 = grad_given_arma;
      }
      for (int i = 0; i < n_given; i++)
      {
        if (marginal_names[g_to_ind(i)] != "normal")
        {
          grad_given_arma.col(i) = grad_given_arma.col(i) % 
                                   given_d_marginal.col(i);
        }
      }
    }
    else
    {
      grad_sum = grad_lower_arma + grad_upper_arma;
      grad_given_arma = grad_sum * (-s12s22);
    }
  }

  // Estimate additional components for variance
  // of conditional distribution when marginal
  // distributions are not normal
  arma::mat grad_var_marginal_d;
  if (is_marginal & grad_sigma)
  {
    for (int i = 0; i < n_given; i++)
    {
      int i_g = g_to_ind.at(i);
      if (marginal_names[i_g] != "normal")
      {
        grad_var_marginal.col(i_g) = q_given.col(i) / 
                                     (2 * sqrt(sigma(i_g, i_g))) -
                                     (given_d_marginal.col(i) % 
                                      given_x_mat0.col(i)) / 
                                     (2 * sigma(i_g, i_g));
      }
    }
  }

  // Estimate gradient respect to marginal distribution parameters
  arma::field<arma::mat> grad_marginal_arma;
  if (is_marginal & grad_marginal)
  {
    grad_marginal_arma = arma::field<arma::mat>(n_dim);
    for (int i = 0; i < n_dim; i++)
    {
      int i_adj;
      grad_marginal_arma.at(i) = arma::mat(n, n_marginal_par[i]);
      if ((marginal_names[i] == "PGN") || (marginal_names[i] == "hpa") ||
          (marginal_names[i] == "student") || (marginal_names[i] == "t"))
      {
        if (given_ind_logical[i])
        {
          i_adj = ind_to_g[i];
          grad_marginal_arma.at(i) = grad_marginal_arma.at(i) +
                                     grad_marginal_list_given.at(i_adj).each_col() %
                                     grad_given_arma0.col(i_adj);
        }
        else
        {
          i_adj = ind_to_d[i];
          grad_marginal_arma.at(i) = grad_marginal_arma.at(i) +
                                     grad_marginal_list_lower.at(i_adj).each_col() %
                                     grad_lower_arma0.col(i_adj);
          grad_marginal_arma.at(i) = grad_marginal_arma.at(i) +
                                     grad_marginal_list_upper.at(i_adj).each_col() %
                                     grad_upper_arma0.col(i_adj);
        }
      }
    }
  }

  // Estimate gradient respect to covariance matrix elements
  // Array to store the partial derivatives
  arma::cube grad_sigma_arma;
  // Main calculations
  if (grad_sigma)
  {
    grad_sigma_arma = arma::cube(n_dim, n_dim, n);
    // Prepare some vectors and matrices
    NumericVector mean_c = NumericVector(2);
    NumericMatrix sigma_c = NumericMatrix(2, 2);
    NumericMatrix upper_ij = NumericMatrix(n, 2);
    List pdf11; List pdf10; List pdf01; List pdf00;
    List cdf11; List cdf10; List cdf01; List cdf00;
    arma::vec cdf11_val(n); arma::vec cdf10_val(n); 
    arma::vec cdf01_val(n); arma::vec cdf00_val(n);
    // Fill with ones for (n_dim == 2) case
    cdf11.fill(1.0); cdf10.fill(1.0); 
    cdf01.fill(1.0); cdf00.fill(1.0);
    // Matrices for lower integration limits
    NumericMatrix lower_c;
    NumericMatrix upper_c;
    // Gradient respect to sigma_d elements
    for (int i = 0; i < n_dependent; i++)
    {
      for (int j = i; j < n_dependent; j++)
      {
        if (i != j)
        {
          // Prepare mean and sigma
          sigma_c(0, 0) = sigma_cond(i, i);
          sigma_c(0, 1) = sigma_cond(i, j);
          sigma_c(1, 0) = sigma_cond(i, j);
          sigma_c(1, 1) = sigma_cond(j, j);
          // Create some matrices
          if (n_dependent > 2)
          {
            lower_c = NumericMatrix(n, n_dependent - 2);
            upper_c = NumericMatrix(n, n_dependent - 2);
            int counter = 0;
            for (int k = 0; k < n_dependent; k++)
            {
              if ((k != i) && (k != j))
              {
                lower_c(_, counter) = lower_d(_, k);
                upper_c(_, counter) = upper_d(_, k);
                counter++;
              }
            }
          }
          // Calculations for 11
          upper_ij(_, 0) = upper_d(_, i);
          upper_ij(_, 1) = upper_d(_, j);
          // Select observations for which
          // this contribution is not zero
          is_use1 = (upper_ij(_, 0) != R_PosInf) &
                    (upper_ij(_, 1) != R_PosInf);
          is_use_n1 = sum(is_use1);
          control_special["is_use"] = is_use1;
          control_special["is_use_n"] = is_use_n1;
          // Estimate the contribution for these observations
          pdf11 = dmnorm(upper_ij,
                         mean_c, sigma_c,
                         NumericVector(), 
                         false, false, false, false,
                         control_special, n_cores);
          arma::vec pdf11_val = pdf11["den"];
          if (n_dependent > 2)
          {
            // Estimate the second part of contribution
            cdf11 = pmnorm(lower_c, upper_c,
                           upper_ij,
                           mean_zero_d, sigma_cond,
                           NumericVector::create(i + 1, j + 1),
                           n_sim, method, ordering, 
                           false, false, false, false, false, false,
                           control_special, n_cores);
            arma::vec cdf11_tmp = cdf11["prob"];
            cdf11_val = cdf11_tmp;
          }
          else
          {
            cdf11_val.fill(1.0);
          }
          // Calculations for 00
          upper_ij(_, 0) = lower_d(_, i);
          upper_ij(_, 1) = lower_d(_, j);
          // Select observations for which
          // this contribution is not zero
          is_use1 = (upper_ij(_, 0) != R_NegInf) &
                    (upper_ij(_, 1) != R_NegInf);
          is_use_n1 = sum(is_use1);
          control_special["is_use"] = is_use1;
          control_special["is_use_n"] = is_use_n1;
          // Estimate the contribution for these observations
          pdf00 = dmnorm(upper_ij,
                         mean_c, sigma_c,
                         NumericVector(), 
                         false, false, false, false,
                         control_special, n_cores);
          arma::vec pdf00_val = pdf00["den"];
          if (n_dependent > 2)
          {
            // Estimate the second part of this contribution
            cdf00 = pmnorm(lower_c, upper_c,
                           upper_ij,
                           mean_zero_d, sigma_cond,
                           NumericVector::create(i + 1, j + 1),
                           n_sim, method, ordering, 
                           false, false, false, false, false, false,
                           control_special, n_cores);
            arma::vec cdf00_tmp = cdf00["prob"];
            cdf00_val = cdf00_tmp;
            }
          else
          {
            cdf00_val.fill(1.0);
          }
          // Calculations for 01
          upper_ij(_, 0) = lower_d(_, i);
          upper_ij(_, 1) = upper_d(_, j);
          // Select observations for which
          // this contribution is not zero
          is_use1 = (upper_ij(_, 0) != R_NegInf) &
                    (upper_ij(_, 1) != R_PosInf);
          is_use_n1 = sum(is_use1);
          control_special["is_use"] = is_use1;
          control_special["is_use_n"] = is_use_n1;
          // Estimate the contribution for these observations
          pdf01 = dmnorm(upper_ij,
                         mean_c, sigma_c,
                         NumericVector(), 
                         false, false, false, false,
                         control_special, n_cores);
          arma::vec pdf01_val = pdf01["den"];
          if (n_dependent > 2)
          {
            // Estimate the second part of this contribution
            cdf01 = pmnorm(lower_c, upper_c,
                           upper_ij,
                           mean_zero_d, sigma_cond,
                           NumericVector::create(i + 1, j + 1),
                           n_sim, method, ordering, 
                           false, false, false, false, false, false,
                           control_special, n_cores);
            arma::vec cdf01_tmp = cdf01["prob"];
            cdf01_val = cdf01_tmp;
          }
          else 
          {
            cdf01_val.fill(1.0);
          }
          // Calculations for 10
          upper_ij(_, 0) = upper_d(_, i);
          upper_ij(_, 1) = lower_d(_, j);
          // Select observations for which
          // this contribution is not zero
          is_use1 = (upper_ij(_, 0) != R_PosInf) &
                    (upper_ij(_, 1) != R_NegInf);
          is_use_n1 = sum(is_use1);
          control_special["is_use"] = is_use1;
          control_special["is_use_n"] = is_use_n1;
          // Estimate the contribution for these observations
          pdf10 = dmnorm(upper_ij,
                         mean_c, sigma_c,
                         NumericVector(), 
                         false, false, false, false,
                         control_special, n_cores);
          arma::vec pdf10_val = pdf10["den"];
          if (n_dependent > 2)
          {
            // Estimate the second part of this contribution
            cdf10 = pmnorm(lower_c, upper_c,
                           upper_ij,
                           mean_zero_d, sigma_cond,
                           NumericVector::create(i + 1, j + 1),
                           n_sim, method, ordering, 
                           false, false, false, false, false, false,
                           control_special, n_cores);
            arma::vec cdf10_tmp = cdf10["prob"];
            cdf10_val = cdf10_tmp;
          }
          else 
          {
            cdf10_val.fill(1.0);
          }
          // Aggregate the result
          int i_d = d_to_ind.at(i);
          int j_d = d_to_ind.at(j);
          arma::vec grad_sigma_ij = pdf11_val % cdf11_val +
                                    pdf00_val % cdf00_val -
                                    pdf01_val % cdf01_val -
                                    pdf10_val % cdf10_val;
          grad_sigma_arma.tube(i_d, j_d) = grad_sigma_ij;
          grad_sigma_arma.tube(j_d, i_d) = grad_sigma_ij;

          // Contribute to variance derivative
          grad_var.col(i_d) = grad_var.col(i_d) -
                              grad_sigma_ij *
                              (sigma_cond_arma.at(i, j) / 
                              (2 * sigma_cond_arma.at(i, i)));

          grad_var.col(j_d) = grad_var.col(j_d) -
                              grad_sigma_ij *
                              (sigma_cond_arma.at(i, j) / 
                              (2 * sigma_cond_arma.at(j, j)));
        }
      }
    }
    
    // Store derivative respect to the variance
    for (int i = 0; i < n_dependent; i++)
    {
      int i_d = d_to_ind.at(i);
      grad_sigma_arma.tube(i_d, i_d) = grad_var.col(i_d);
    }

    // Gradient respect to sigma_dg and sigma_gg
    arma::mat grad_var0;
    if (is_marginal)
    {
      grad_var0 = grad_var - grad_var_marginal;
    }
    if (n_given > 0)
    {
      diff_mean_by_sigma = -diff_mean_by_sigma;
      // respect to sigma_dg elements
      for (int i = 0; i < n_dependent; i++)
      {
        for (int j = 0; j < n_given; j++)
        {
          int i_d = d_to_ind(i);
          int j_g = g_to_ind(j);
          // part associated with conditional mean
          if (is_marginal)
          {
            grad_sigma_arma.tube(i_d, j_g) = (grad_upper_arma0.col(i) +  
                                              grad_lower_arma0.col(i)) % 
                                              diff_mean_by_sigma.col(j);
          }
          else
          {
            grad_sigma_arma.tube(i_d, j_g) = (grad_upper_arma.col(i) +  
                                              grad_lower_arma.col(i)) % 
                                              diff_mean_by_sigma.col(j);
          }
            for (int j1 = 0; j1 < n_dependent; j1++)
            {
              // part associated with conditional covariance
              if (is_marginal)
              {
                if (i == j1)
                {
                  arma::vec grad_sigma_ij = grad_sigma_arma.tube(i_d, j_g);
                  grad_sigma_arma.tube(i_d, j_g) = 
                    grad_sigma_ij - (2 * s12s22.at(j1, j)) * grad_var0.col(i_d);
                }
                else
                {
                  grad_sigma_arma.tube(i_d, j_g) =
                    grad_sigma_arma.tube(i_d, j_g) -
                    s12s22.at(j1, j) * 
                    grad_sigma_arma.tube(i_d, d_to_ind.at(j1));
                }
              }
              else
              {
                grad_sigma_arma.tube(i_d, j_g) =
                  grad_sigma_arma.tube(i_d, j_g) -
                  ((1 + (i == j1)) * s12s22.at(j1, j)) *
                  grad_sigma_arma.tube(i_d, d_to_ind.at(j1));
              }
            }
            grad_sigma_arma.tube(j_g, i_d) = grad_sigma_arma.tube(i_d, j_g);
        }
      }
      // respect to sigma_g elements
      diff_mean_by_sigma = -diff_mean_by_sigma;
      arma::mat sigma_dg = cond["sigma_dg"];
      arma::mat sigma_g_inv = cond["sigma_g_inv"];
      if (is_marginal)
      {
        grad_var_marginal_d = arma::mat(n, n_dependent);
      }
      for (int i = 0; i < n_given; i++)
      {
        for (int j = i; j < n_given; j++)
        {
          arma::mat I_g = arma::mat(n_given, n_given, arma::fill::zeros);
          I_g.at(i, j) = 1;
          I_g.at(j, i) = 1;
          arma::mat mat_tmp = sigma_dg * sigma_g_inv * I_g * 
                              sigma_g_inv * sigma_dg.t();
          arma::mat mat_tmp2 = given_x_arma * 
                               (sigma_dg * sigma_g_inv * I_g * sigma_g_inv).t();
          // part associated with conditional mean
          arma::mat mat_tmp3 = sum(mat_tmp2 % grad_sum, 1);
          if (is_marginal)
          {
            if ((i == j) & (marginal_names[g_to_ind(i)] != "normal"))
            {
              for (int t = 0; t < n_dependent; t++)
              {
                mat_tmp3 = mat_tmp3 - (grad_var_marginal.col(g_to_ind(i)) %
                                       grad_sum.col(t)) * s12s22.at(t, i);
              }
            }
          }
          for (int i1 = 0; i1 < n_dependent; i1++)
          {
            for (int j1 = i1; j1 < n_dependent; j1++)
            {
              // part associated with conditional covariance
              arma::colvec mat_tmp4;
              if (is_marginal & (i1 == j1))
              {
                mat_tmp4 = grad_var0.col(d_to_ind[i1]);
              }
              else
              {
                mat_tmp4 = grad_sigma_arma.tube(d_to_ind[i1], 
                                                d_to_ind[j1]);
              }
              mat_tmp3 = mat_tmp3 + mat_tmp.at(i1, j1) * mat_tmp4;
            }
          }
          grad_sigma_arma.tube(g_to_ind[i], g_to_ind[j]) = mat_tmp3;
          grad_sigma_arma.tube(g_to_ind[j], g_to_ind[i]) = mat_tmp3;
        }
      }
    }
  }

  // Deal with logarithm respect to gradients
  if (log)
  {
    // for lower
    if (grad_lower)
    {
      for (int i = 0; i < n_dependent; i++)
      {
        grad_lower_arma.col(i) = grad_lower_arma.col(i) / prob;
      }
    }
    // for upper
    if (grad_upper)
    {
      for (int i = 0; i < n_dependent; i++)
      {
        grad_upper_arma.col(i) = grad_upper_arma.col(i) / prob;
      }
    }
    // for conditioned values
    if (grad_given)
    {
      for (int i = 0; i < n_given; i++)
      {
        grad_given_arma.col(i) = grad_given_arma.col(i) / prob;
      }
    }
    // for sigma
    if (grad_sigma)
    {
      for (int i = 0; i < n_dim; i++)
      {
        for (int j = 0; j < n_dim; j++)
        {
          arma::vec mat_tmp = grad_sigma_arma.tube(i, j);
          grad_sigma_arma.tube(i, j) = mat_tmp / prob;
        }
      }
    }
    // for marginal
    if (grad_marginal & is_marginal)
    {
      for (int i = 0; i < n_dim; i++)
      {
        if (n_marginal_par[i] > 0)
        {
          grad_marginal_arma.at(i) = grad_marginal_arma.at(i).each_col() / prob;
        }
      }
      if (grad_marginal_prob)
      {
        grad_upper_arma0 = grad_upper_arma0.each_col() / prob;
        grad_lower_arma0 = grad_lower_arma0.each_col() / prob;
        if (n_given > 0)
        {
          grad_given_arma0 = grad_given_arma0.each_col() / prob;
        }
      }
    }
  }
  // Store the gradients
  if (grad_upper)
  {
    return_list["grad_upper"] = grad_upper_arma;
  }
  if (grad_lower)
  {
    return_list["grad_lower"] = grad_lower_arma;
  }
  if (grad_given)
  {
    return_list["grad_given"] = grad_given_arma;
  }
  if (grad_sigma)
  {
    return_list["grad_sigma"] = grad_sigma_arma;
  }
  if (grad_marginal)
  {
    if (grad_marginal_prob)
    {
      return_list["grad_upper_marginal"] = grad_upper_arma0;
      return_list["grad_lower_marginal"] = grad_lower_arma0;
      return_list["grad_given_marginal"] = grad_given_arma0;
    }
    return_list["grad_marginal"] = grad_marginal_arma;
  }
  // Return the results
  return_list.attr("class") = "mnorm_pmnorm";
  return(return_list);
}

// [[Rcpp::export(rng = false)]]
double GHK(const NumericVector lower, 
           const NumericVector upper, 
           const NumericMatrix sigma, 
           const arma::mat h,
           const String ordering = "default",
           const int n_sim = 1000,
           const int n_cores = 1)
{
  // Get dimensions of the distribution
  int n_dim = lower.size();
  
  // Perform Cholesky decomposition of
  // the covariance matrix
  arma::mat L = chol(as<arma::mat>(sigma), "lower");  
  
  // Convert integration limits to arma
  arma::vec lwr = as<arma::vec>(lower);
  arma::vec upr = as<arma::vec>(upper);
  
  // Perform variables reordering
  if (ordering != "NO")
  {
    // Convert covariance matrix to arma
    arma::mat S = as<arma::mat>(sigma);
    arma::vec S_diag_sqrt = arma::sqrt(arma::diagvec(S));
    
    // Adjust for infinity values
    arma::vec lwr_adj = 1 * lwr;
    arma::vec upr_adj = 1 * upr;
    for (int i = 0; i < n_dim; i++)
    {
      if (!std::isfinite(lwr_adj.at(i)))
      {
        lwr_adj.at(i) = -S_diag_sqrt.at(i) * 10;
      }
      if (!std::isfinite(upr_adj.at(i)))
      {
        upr_adj.at(i) = S_diag_sqrt.at(i) * 10;
      }
    }
    
    // Adjust lower and upper integration limits
    arma::vec a0 = lwr_adj / S_diag_sqrt;
    arma::vec b0 = upr_adj / S_diag_sqrt;
    
    // Adjust infinity values
    for (int i = 0; i < n_dim; i++)
    {
      if (!std::isfinite(a0.at(i)))
      {
        a0.at(i) = -S_diag_sqrt.at(i) * 10;
      }
      if (!std::isfinite(b0.at(i)))
      {
        b0.at(i) = S_diag_sqrt.at(i) * 10;
      }
    }
    
    // Vector to store values determining the ordering
    arma::vec min_val;
    
    // Vector to store truncated expectations
    arma::colvec tr_exp = arma::colvec(n_dim - 2);
    arma::vec tr_exp_tmp;
    
    // Integers to store minimum value for each iteration
    int min_ind_adj;
    int min_ind;
    
    // Temporal vector
    arma::vec vec_tmp = arma::regspace(1, n_dim);

    // Ordering routine
    for (int i = 0; i < (n_dim - 1); i++)
    {
      // Adjust integration limits prior to minimization routine
      if (i > 0)
      {
        a0 = arma::vec(n_dim - i);
        b0 = arma::vec(n_dim - i);
        for (int j = i; j < n_dim; j++)
        {
          arma::vec tr_exp_comb = L.submat(j, 0, j, i - 1) * 
                                  tr_exp.subvec(0, i - 1);
          arma::vec adj = sqrt(S.at(j, j) - L.submat(j, 0, j, i - 1) *
                                            L.submat(j, 0, j, i - 1).t());
          a0.at(j - i) = (lwr_adj.at(j) - tr_exp_comb.at(0)) / adj.at(0);
          b0.at(j - i) = (upr_adj.at(j) - tr_exp_comb.at(0)) / adj.at(0);
        }
      }
      // Calculate preliminary values
      arma::vec p_a0 = arma::normcdf(a0);
      arma::vec p_b0 = arma::normcdf(b0);
      arma::vec d_a0 = arma::normpdf(a0);
      arma::vec d_b0 = arma::normpdf(b0);
      arma::vec p_diff0 = p_b0 - p_a0;
      tr_exp_tmp = (d_a0 - d_b0) / p_diff0;
      
      // Estimate minimization criteria value
      if (ordering == "mean")
      {
        min_val = p_diff0;
      }
      
      if (ordering == "variance")
      {
        min_val = ((a0 % d_a0 - b0 % d_b0) / p_diff0) - 
                  arma::pow(tr_exp_tmp, 2);
      }
      
      // Select minimum value
      min_ind_adj = index_min(min_val);
      min_ind = min_ind_adj + i;
      
      // Swap the elements of integration limits,
      // covariance matrix and Cholesky decomposition
      if (min_ind != i)
      {
        S.swap_rows(i, min_ind);
        S.swap_cols(i, min_ind);
        L = chol(S, "lower");
        lwr.swap_rows(i, min_ind);
        upr.swap_rows(i, min_ind);
        lwr_adj.swap_rows(i, min_ind);
        upr_adj.swap_rows(i, min_ind);
        vec_tmp.swap_rows(i, min_ind);
      }
      
      // Estimate truncated mean
      if(i < (n_dim - 2))
      {
        tr_exp.at(i) = tr_exp_tmp.at(min_ind_adj);
      }
    }
  }

  // Initialize vectors to store lower and upper bounds
  // for simulation from truncated normal
  arma::vec a = arma::vec(n_sim, 
                          arma::fill::value(arma::normcdf(lwr(0) / L(0, 0))));
  arma::vec b = arma::vec(n_sim, 
                          arma::fill::value(arma::normcdf(upr(0) / L(0, 0))));

  // Initialize vectors to store realizations
  // of truncated normal variables
  arma::mat z_tr = arma::mat(n_sim, n_dim - 1);
  
  // Vector to store preliminary values
  arma::vec inner_tmp = arma::vec(n_sim);
  
  // Vector to store probabilities
  arma::vec prob = arma::vec(n_sim, arma::fill::value(b(0) - a(0)));
  
  // Vector to store differences
  arma::vec ab_diff = arma::vec(n_sim, arma::fill::value(b(0) - a(0)));
  
  // Start the main procedure
  arma::vec vec_tmp(n_sim);
  for (int i = 0; i < n_dim; i++)
  {
    if(i > 0)
    {
      arma::vec inner_tmp = z_tr.cols(0, i - 1) * L.submat(i, 0, i, i - 1).t();
      a = arma::normcdf((lwr.at(i) - inner_tmp) / L.at(i, i));
      b = arma::normcdf((upr.at(i) - inner_tmp) / L.at(i, i));
      ab_diff = b - a;
      prob = prob % ab_diff;
    }

    if (i < (n_dim - 1))
    {
      vec_tmp = a + h.submat(0, i, n_sim - 1, i) % ab_diff;
      z_tr.col(i) = qnormFast(vec_tmp, 0, 1, "Voutier", false, n_cores);
      //NumericVector vec_tmp1 = wrap(vec_tmp);
      //vec_tmp1 = Rcpp::Rcpp::qnorm(vec_tmp1, 0.0, 1.0);
      //z_tr.col(i) = as<arma::vec>(vec_tmp1);
    }
  }
  
  // Calculate the result controlling 
  // for possible NaN values
  double prob_val;
  if (prob.has_nan())
  {
    NumericVector prob_new = wrap(prob);
    prob_new = na_omit(prob_new);
    prob_val = Rcpp::mean(prob_new);
  }
  else 
  {
    prob_val = arma::mean(prob);
  }
  
  return(prob_val);
}
