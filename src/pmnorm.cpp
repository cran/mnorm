#include <RcppArmadillo.h>
#include "halton.h"
#include "pmnorm.h"
#include "dmnorm.h"
#include "GaussLegendre.h"
#include "cmnorm.h"
#include "qmnorm.h"
using namespace Rcpp;

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
//' @template return_pmnorm_Template
//' @template example_pmnorm_Template
//' @references Genz, A. (2004), Numerical computation of rectangular bivariate 
//' and trivariate normal and t-probabilities, Statistics and 
//' Computing, 14, 251-260.
//' @references Genz, A. and Bretz, F. (2009), Computation of Multivariate 
//' Normal and t Probabilities. Lecture Notes in Statistics, Vol. 195. 
//' Springer-Verlag, Heidelberg.
//' @references E. Kossova., B. Potanin (2018). 
//' Heckman method and switching regression model multivariate generalization.
//' Applied Econometrics, vol. 50, pages 114-143.
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
            Nullable<List> control = R_NilValue,
            const int n_cores = 1)
{
  // Multiple cores
  omp_set_num_threads(n_cores);
  
  // Create output list
  List return_list;
  
  // Check whether any gradients should be calculated
  const bool is_grad = (grad_lower | grad_upper | grad_sigma | grad_given);
  
  // Get number of dimensions
  const int n_dim = sigma.nrow();
  
  // Get the number of conditioned and unconditioned components
  const int n_given = given_ind.size();
  const int n_dependent = n_dim - n_given;
  
  // Get number of observations
  const int n = lower.size() / n_dependent;
  
  // Provide input validation if need
  if (is_validation)
  {
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
      "and 'all(!is.nan(given_ind)).'";
      stop(stop_message);
    }
    
    if (unique(given_ind).size() != given_ind.size())
    {
      std::string stop_message = "Duplicates have been found in 'given_ind'. "
      "Please, insure that 'length(unique(given_ind)) == length(given_ind)'.";
      stop(stop_message);
    }
    
    if (!as<arma::mat>(sigma).is_sympd())
    {
      std::string stop_message = "Not positively definite covariance matrix. "
      "Please, insure that 'sigma' is positively definite covariance matrix.";
      stop(stop_message);
    }
    
    if ((ordering != "NO") & (ordering != "mean") & (ordering != "variance"))
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
                                  R_NilValue, n_cores);
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
  given_ind_logical[given_ind - 1] = true;
  dependent_ind = ind[!given_ind_logical];
  
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
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n_dependent; j++)
    {
      int j_d = d_to_ind(j);
      lower_mat(i, j) = lower_mat(i, j) - mean[j_d];
      upper_mat(i, j) = upper_mat(i, j) - mean[j_d];
    }
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
    // Adjust for zero mean
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n_given; j++)
      {
        int j_g = g_to_ind(j);
        given_x_mat(i, j) = given_x_mat(i, j) - mean[j_g];
      }
    }
    
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
  
  
  //Special routine for univariate case
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
  if (n_dependent == 2)
  {
    // Preliminary constants
    const double pi = 3.141592653589793238463;
    
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
    
    // Deal with infinite upper integration limits
    upr1.replace(arma::datum::inf, 10);
    upr2.replace(arma::datum::inf, 10);
    
    // Preliminary value for vectorization purposes
    arma::vec adj = 1 - pow(x, 2);
    arma::vec adj1 = 1 / (-2 * adj);
    arma::vec adj2 = w % (rho / (4 * pi * sqrt(adj)));

    // Estimate probabilities P(X1 < upper1, X2 < upper2)
    prob = pmnorm2(upr1, upr2, x, adj, adj1, adj2, n_cores);

    // Additional routine if lower limits are
    // not negative infinite
    bool lower1_inf = all(is_infinite(lower_d(_, 0)));
    bool lower2_inf = all(is_infinite(lower_d(_, 1)));

    if (!lower1_inf)
    {
      lwr1.replace(-arma::datum::inf, -10);
      prob = prob - pmnorm2(lwr1, upr2, x, adj, adj1, adj2, n_cores);
    }
    
    if (!lower2_inf)
    {
      lwr2.replace(-arma::datum::inf, -10);
      prob = prob - pmnorm2(lwr2, upr1, x, adj, adj1, adj2, n_cores);
    }
    
    if ((!lower1_inf) & (!lower2_inf))
    {
      prob = prob + pmnorm2(lwr1, lwr2, x, adj, adj1, adj2, n_cores);
    }
  }
  
  // Special routine for multivariate case (at least 3 dimensions)
  if (n_dependent > 2)
  {
    // Generate Halton sequence
    arma::mat h(as<arma::mat>(halton(n_sim, seqPrimes(n_dim), 100,
                                     "NO", "halton", false, n_cores)));
    h.reshape(n_sim, n_dim);

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
  
  // Subtract one from indexes and transform
  // them into arma format
  arma::uvec dependent_arma = as<arma::uvec>(dependent_ind) - 1;
  arma::uvec given_arma = as<arma::uvec>(given_ind) - 1;
  
  // Create vector of zero means
  NumericVector mean_zero_d = NumericVector(n_dependent);
  
  // Vector of conditional standard deviations
  arma::vec sd_cond_arma = arma::sqrt(arma::diagvec(sigma_cond_arma));
  
  // Conver matrix of conditioned values to arma
  arma::mat const given_x_arma(given_x_mat.begin(), n, n_given, false);
  
  // Matrix to store gradient for variances
  arma::mat grad_var(n, n_dim);
  
  // Variable to control for observations need to calculate
  // appropriate parts of Jacobian
  LogicalVector is_use1;
  int is_use_n1;

  // Estimate gradient respect to lower integration limits
  arma::mat grad_lower_arma(n, n_dependent);
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
    // Contribute to variance derivative
    if (grad_sigma)
    {
      arma::vec lwr_adj_tmp = -lower_d_arma.col(i) /
                               (2 * sigma_cond_arma.at(i, i));
      lwr_adj_tmp.replace(arma::datum::inf, 0);
      int i_d = d_to_ind.at(i);
      grad_var.col(i_d) = grad_var.col(i_d) + 
                          grad_lower_arma.col(i) % 
                          lwr_adj_tmp;
    }
  }

  // Estimate gradient respect to upper integration limits
  arma::mat grad_upper_arma(n, n_dependent);
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
    // Contribute to variance derivative
    if (grad_sigma)
    {
      arma::vec upr_adj_tmp = -upper_d_arma.col(i) / 
                              (2 * sigma_cond_arma.at(i, i));
      upr_adj_tmp.replace(-arma::datum::inf, 0);
      int i_d = d_to_ind.at(i);
      grad_var.col(i_d) = grad_var.col(i_d) + 
                          grad_upper_arma.col(i) % 
                          upr_adj_tmp;
    }
  }

  // Estimate the gradient respect to conditional values
  arma::mat grad_given_arma;
  if (grad_given)
  {
    grad_given_arma = (grad_lower_arma + grad_upper_arma) * (-s12s22);
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
              if ((k != i) & (k != j))
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
          grad_sigma_arma.tube(i_d, j_g) = (grad_upper_arma.col(i) +  
                                            grad_lower_arma.col(i)) % 
                                            diff_mean_by_sigma.col(j);
            for (int j1 = 0; j1 < n_dependent; j1++)
            {
              // part associated with conditional covariance
              grad_sigma_arma.tube(i_d, j_g) =
                grad_sigma_arma.tube(i_d, j_g) -
                (1 + (i == j1)) * s12s22.at(j1, j) *
                grad_sigma_arma.tube(i_d, d_to_ind.at(j1));
            }
            grad_sigma_arma.tube(j_g, i_d) = grad_sigma_arma.tube(i_d, j_g);
        }
      }
      diff_mean_by_sigma = -diff_mean_by_sigma;
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
          arma::mat mat_tmp2 = given_x_arma * 
                               (sigma_dg * sigma_g_inv * I_g * sigma_g_inv).t();
          // part associated with conditional mean
          arma::mat mat_tmp3 = sum(mat_tmp2 % (grad_upper_arma + 
                                               grad_lower_arma), 1);
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
  
  // Return the results
  return_list.attr("class") = "mnorm_pmnorm";
  return(return_list);
}

// [[Rcpp::export(rng = false)]]
arma::vec pmnorm2(const arma::vec x1,
                  const arma::vec x2,
                  const arma::vec x,
                  const arma::vec adj,
                  const arma::vec adj1,
                  const arma::vec adj2,
                  const int n_cores = 1)
{
  // Multiple cores
  omp_set_num_threads(n_cores);
  
  // Get number of probabilities to calculate
  int n = x1.size();
  
  // Create vector to store the results
  arma::vec prob(n);
  
  // Some preliminary values
  arma::vec x1_prob = arma::normcdf(x1);
  arma::vec x2_prob = arma::normcdf(x2);
  arma::vec x1_pow = pow(x1, 2.0);
  arma::vec x2_pow = pow(x2, 2.0);
  arma::vec x1x2_prob = x1_prob % x2_prob;
  arma::vec x1x2_sum = x1_pow + x2_pow;
  arma::vec x1x2_prod = 2 * x1 % x2;
  
  // Estimation of probabilities for each observation
  for(int i = 0; i < n; i++)
  {
    prob.at(i) = x1x2_prob.at(i) +
    sum(adj2 % exp((x1x2_sum.at(i) - x1x2_prod.at(i) * x) % adj1));
  }
  
  return(prob);
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
  // Multiple cores
  omp_set_num_threads(n_cores);
  
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
    
    // Integer to store minimum value for each iteration
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
      //vec_tmp1 = Rcpp::qnorm(vec_tmp1, 0.0, 1.0);
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
