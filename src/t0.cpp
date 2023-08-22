#define ARMA_DONT_USE_OPENMP
#include <RcppArmadillo.h>
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

//' Differentiate Regularized Incomplete Beta Function.
//' @description Calculate derivatives of the regularized incomplete 
//' beta function that is a cumulative distribution function of the beta
//' distribution.
//' @param x numeric vector of values between 0 and 1. It is similar to
//' \code{q} argument of \code{\link[stats]{pbeta}} function.
//' @param p similar to \code{shape1} argument of 
//' \code{\link[stats]{pbeta}} function.
//' @param q similar to \code{shape2} argument of 
//' \code{\link[stats]{pbeta}} function.
//' @param n positive integer representing the number of iterations used
//' to calculate the derivatives. Greater values provide higher accuracy by the
//' cost of more computational resources.
//' @param is_validation logical; if \code{TRUE} then input arguments are
//' validated. Set to \code{FALSE} to slightly increase the performance
//' of the function.
//' @param control list of control parameters. Currently not intended 
//' for the users.
//' @details The function implements differentiation algorithm of 
//' R. Boik and J. Robinson-Cox (1998). 
//' Currently only first-order derivatives are considered.
//' @return The function returns a list which has the following elements:
//' \itemize{
//' \item \code{dx} - numeric vector of derivatives respect to each 
//' element of \code{x}.
//' \item \code{dp} - numeric vector of derivatives respect to \code{p} for
//' each element of \code{x}.
//' \item \code{dq} - numeric vector of derivatives respect to \code{q} for
//' each element of \code{x}.
//' }
//' @references Boik, R. J. and Robinson-Cox, J. F. (1998). Derivatives of the 
//' Incomplete Beta Function. Journal of Statistical Software, 3 (1),
//' pages 1-20.
//' @template example_pbetaDiff_Template
// [[Rcpp::export(rng = false)]]
List pbetaDiff(const arma::vec x,
               const double p = 10,
               const double q = 0.5,
               const int n = 10,
               const bool is_validation = true,
               const Nullable<List> control = R_NilValue)
{
  // Validate input if need
  if (is_validation)
  {
    if (p <= 0)
    {
      stop ("Argument 'p' should be positive.");
    }
    if (q <= 0)
    {
      stop ("Argument 'q' should be positive.");
    }
    if (any(x <= 0))
    {
      stop ("All elements of 'x' should be positive.");
    }
    if (any(x >= 1))
    {
      stop ("All elements of 'x' should be less than one.");
    }
    if (n <= 0)
    {
      stop ("Argument 'n' should be a positive integer.");
    }
  }
  
  // Prepare output list
  List out;
  
  // Deal with control parameters
  List control1(control);
  bool is_recursion = true;
  if (control != R_NilValue)
  {
    if (control1.containsElementNamed("is_recursion"))
    {
      bool is_recursion_tmp = control1["is_recursion"];
      is_recursion = is_recursion_tmp;
    }
    else
    {
      is_recursion = false;
      control1["is_recursion"] = is_recursion;
    }
  }
  
  // Get the number of observations
  const int n_x = x.size();
  
  // Use recursion if need
  if (is_recursion)
  {
    const double cond = p / (p + q);
    arma::uvec ind_lower = arma::find(x >= cond);
    arma::uvec ind_upper = arma::find(x < cond);
    const int n_lower = ind_lower.size();
    const int n_upper = ind_upper.size();
    if (n_lower != 0)
    {
      // Prepare the vector for output
      arma::vec dx_total(n_x);
      arma::vec dp_total(n_x);
      arma::vec dq_total(n_x);
      
      // Deal with lower
      arma::vec x_lower = x.elem(ind_lower);
      List out_lower = pbetaDiff(1 - x_lower, q, p, n, false, control1);
      
      arma::vec dx_lower = out_lower["dx"];
      arma::vec dp_lower = out_lower["dp"];
      arma::vec dq_lower = out_lower["dq"];
      
      dx_total.elem(ind_lower) = dx_lower;
      dp_total.elem(ind_lower) = -dq_lower;
      dq_total.elem(ind_lower) = -dp_lower;
      
      // Deal with upper
      if (n_upper != 0)
      {
        arma::vec x_upper = x.elem(ind_upper);
        List out_upper = pbetaDiff(x_upper, p, q, n, false, control1);
        
        arma::vec dx_upper = out_upper["dx"];
        arma::vec dp_upper= out_upper["dp"];
        arma::vec dq_upper = out_upper["dq"];
        
        dx_total.elem(ind_upper) = dx_upper;
        dp_total.elem(ind_upper) = dp_upper;
        dq_total.elem(ind_upper) = dq_upper;
      }
      
      // Aggregate the output
      out["dx"] = dx_total;
      out["dp"] = dp_total;
      out["dq"] = dq_total;
      
      // Return the result
      return(out);
    }
  }

  // Calculate a derivative respect to x
  arma::vec dx = (pow(x, p - 1) % pow(1 - x, q - 1)) / R::beta(p, q);
  
  // Calculate K and its derivatives
  arma::vec K = (dx % x) / p;
  double digamma_pq = R::digamma(p + q);
  arma::vec dKdp = K % (log(x) - ((1 / p) - digamma_pq + R::digamma(p)));
  arma::vec dKdq = K % (log(1 - x) + (digamma_pq - R::digamma(q)));
  
  // Special values involving x
  arma::vec f = (q * x) / (p * (1 - x));
  arma::vec f2 = pow(f, 2);
  arma::vec pf = p * f;
  arma::vec pf2q = pf + 2 * q;

  // Special values involving p and q
  const double p2 = pow(p, 2);
  const double q2 = pow(q, 2);
  const double p3 = pow(p, 3);
  const double p4 = pow(p, 4);
  
  // Small letter matrices
  arma::mat a(n_x, n);
  arma::mat b(n_x, n);
  a.col(0) = pf * (q - 1) / (q * (p + 1));
  if (n >= 2)
  {
    for (int i = 1; i < n; i++)
    {
      a.col(i) = f2 * 
                 ((p2 * i * (p + q + i - 1) * (p + i) * (q - i - 1)) /
                  (q2 * (p + 2 * i - 1) * pow(p + 2 * i, 2) * (p + 2 * i + 1)));
    }
  }
  for (int i = 0; i < n; i++)
  {
    b.col(i) = (pf2q * (2 * pow(i + 1, 2) + 2 * (p - 1) * (i + 1)) + 
                (p * q) * ((p - 2) - pf)) /
               (q * (p + 2 * i) * (p + 2 * i + 2));
  }

  // Derivatives of small letter matrices respect to p
  arma::mat dadp(n_x, n);
  dadp.col(0) = pf * ((1 - q) / (q * pow(1 + p, 2)));
  if (n >= 2)
  {
    for (int i = 1; i < n; i++)
    {
      dadp.col(i) = f2 * 
        (-i * p2 * (q - i - 1) *
        ((-8 + 8 * p + 8 * q) * pow(i + 1, 3) + 
         (16 * p2 + (-44 + 20 * q) * p + 26 - 24 * q) * pow(i + 1, 2) +
         (10 * p3 + (14 * q - 46) * p2 + (-40 * q + 66) * p - 28 + 24 * q) * 
         (i + 1) +
         (2 * p4 + (-13 + 3 * q) * p3 + (-14 * q + 30) * p2) +
         (-29 + 19 * q) * p + 10 - 8 * q) / 
        (q2 * pow(p + 2 * i - 1, 2) * pow(p + 2 * i, 3) * 
         pow(p + 2 * i + 1, 2)));
    }
  }
  arma::mat dbdp(n_x, n);
  for (int i = 0; i < n; i++)
  {
    dbdp.col(i) = pf *
                  ((-4 * p - 4 * q + 4) * pow(i + 1, 2) + 
                   (4 * p - 4 + 4 * q - 2 * p2) * (i + 1) + p2 * q) / 
                  (q * pow(p + 2 * i, 2) * pow(p + 2 * i + 2, 2));
  }

  // Derivatives of small letter matrices respect to q
  arma::mat dadq(n_x, n);
  dadq.col(0) = pf / (q * (p + 1));
  if (n >= 2)
  {
    for (int i = 1; i < n; i++)
    {
      dadq.col(i) = f2 * 
        ((p2 * i * (p + i) * (2 * q + p - 2)) /
         (q2 * (p + 2 * i - 1) * pow(p + 2 * i, 2) * (p + 2 * i + 1)));
    }
  }
  arma::mat dbdq(n_x, n);
  for (int i = 0; i < n; i++)
  {
    dbdq.col(i) = pf * (-p / (q * (p + 2 * i) * (p + 2 * i + 2)));
  }

  // Big letter matrices
  arma::mat A(n_x, n + 2);
  A.col(0).fill(1);
  A.col(1).fill(1);
  arma::mat B(n_x, n + 2);
  B.col(0).fill(0);
  B.col(1).fill(1);
  for (int i = 0; i < n; i++)
  {
    A.col(i + 2) = a.col(i) % A.col(i) + b.col(i) % A.col(i + 1);
    B.col(i + 2) = a.col(i) % B.col(i) + b.col(i) % B.col(i + 1);
  }

  // Derivatives of big letter matrices respect to p
  arma::mat dAdp(n_x, n + 2);
  arma::mat dBdp(n_x, n + 2);
  for (int i = 0; i < n; i++)
  {
    dAdp.col(i + 2) = dadp.col(i) % A.col(i) + a.col(i) % dAdp.col(i) + 
                      dbdp.col(i) % A.col(i + 1) + b.col(i) % dAdp.col(i + 1);
    dBdp.col(i + 2) = dadp.col(i) % B.col(i) + a.col(i) % dBdp.col(i) + 
                      dbdp.col(i) % B.col(i + 1) + b.col(i) % dBdp.col(i + 1);
  }
  
  // Derivatives of big letter matrices respect to q
  arma::mat dAdq(n_x, n + 2);
  arma::mat dBdq(n_x, n + 2);
  for (int i = 0; i < n; i++)
  {
    dAdq.col(i + 2) = dadq.col(i) % A.col(i) + a.col(i) % dAdq.col(i) + 
                      dbdq.col(i) % A.col(i + 1) + b.col(i) % dAdq.col(i + 1);
    dBdq.col(i + 2) = dadq.col(i) % B.col(i) + a.col(i) % dBdq.col(i) + 
                      dbdq.col(i) % B.col(i + 1) + b.col(i) % dBdq.col(i + 1);
  }
  
  // Main derivatives
  arma::vec AB = A.col(n + 1) / B.col(n + 1);
  arma::vec AB2 = AB / B.col(n + 1);
  arma:: vec dp = dKdp % AB + 
                  K % (dAdp.col(n + 1) / B.col(n + 1) - dBdp.col(n + 1) % AB2);
  arma:: vec dq = dKdq % AB + 
                  K % (dAdq.col(n + 1) / B.col(n + 1) - dBdq.col(n + 1) % AB2);
  
  // Aggregate the output
  out["dx"] = dx;
  out["dp"] = dp;
  out["dq"] = dq;
  
  // Return the result
  return(out);
}

//' Standardized Student t Distribution
//' @name stdt
//' @description These functions calculate and differentiate a cumulative 
//' distribution function and density function of the standardized 
//' (to zero mean and unit variance) Student distribution. Quantile function 
//' and random numbers generator are also provided.
//' @param x numeric vector of quantiles.
//' @param df positive real value representing the number of degrees of freedom.
//' Since this function deals with standardized Student distribution, argument
//' \code{df} should be greater than \code{2} because otherwise variance is
//' undefined.
//' @param log logical; if \code{TRUE} then probabilities (or densities) p 
//' are given as log(p) and derivatives will be given respect to log(p).
//' @param grad_x logical; if \code{TRUE} then function returns a derivative
//' respect to \code{x}.
//' @param grad_df logical; if \code{TRUE} then function returns a derivative
//' respect to \code{df}.
//' @param n positive integer. If \code{rt0} function is used then this 
//' argument represents the number of random draws. Otherwise \code{n} states 
//' for the number of iterations used to calculate the derivatives associated 
//' with \code{pt0} function via \code{\link[mnorm]{pbetaDiff}} function.
//' @template details_t0_Template
//' @template return_t0_Template
//' @template example_t0_Template
// [[Rcpp::export(rng = false)]]
List dt0(const arma::vec x,
         const double df = 10,
         const bool log = false,
         const bool grad_x = false,
         const bool grad_df = false)
{
   // Validation
   if (df <= 2)
   {
     stop("Argument 'df' should be greater than 2.");
   }
   
   // Prepare output list
   List out;
   
   // Adjust the values
   arma::vec x2 = pow(x, 2);
   arma::vec x2_adj = x2 / (df - 2);
   arma::vec x2_adj2 = x2_adj + 1;
   
   // Estimate density value
   double val = R::gammafn((df + 1) / 2) / 
                (sqrt((df - 2) * arma::datum::pi) * R::gammafn(df / 2));
   arma::vec den = val * pow(x2_adj2, -(df + 1) / 2);
   
   // Estimate common values for the derivatives
   arma::vec x2_adj32;
   if (grad_x || grad_df)
   {
     x2_adj32 = (df - 2) + x2;
   }
   
   // Calculate log-derivative respect to x
   arma::vec grad_x_val;
   if (grad_x)
   {
     grad_x_val = (-(df + 1)) *  (x / x2_adj32);
   }
   
   // Calculate log-derivative respect to df
   arma::vec x2_adj2_log;
   arma::vec grad_df_val;
   if (grad_df)
   {
     x2_adj2_log = arma::log(x2_adj2);
     double dval = 0.5 * (1 / (2 - df) - R::digamma(df / 2) + 
                          R::digamma((df + 1) / 2));
     grad_df_val = dval + 0.5 * (((df + 1) / x2_adj32) % x2_adj - x2_adj2_log);
   }
   
   // Deal with logarithm
   if (log)
   {
     if (grad_df)
     {
       den = x2_adj2_log * (-(df + 1) / 2) + std::log(val);
     }
     else
     {
       den = arma::log(den);
     }
   }
   else
   {
     if (grad_x)
     {
       grad_x_val = grad_x_val % den;
     }
     if (grad_df)
     {
       grad_df_val = grad_df_val % den;
     }
   }
   
   // Aggregate the output
   out["den"] = den;
   if (grad_x)
   {
     out["grad_x"] = grad_x_val;
   }
   if (grad_df)
   {
     out["grad_df"] = grad_df_val;
   }
   
   // Return the result
   return(out);
 }

//' @name stdt
//' @export
// [[Rcpp::export(rng = false)]]
List pt0(const arma::vec x,
         const double df = 10,
         const bool log = false,
         const bool grad_x = false,
         const bool grad_df = false,
         const int n = 10)
{
  // Validation
  if (df <= 2)
  {
    stop ("Argument 'df' should be greater than 2.");
  }
  if (n <= 0)
  {
    stop ("Argument 'n' should be a positive integer.");
  }

  // Control for zero values
  arma::uvec zero_ind = arma::find(x == 0);
  const int n_zero = zero_ind.size();
  
  // Create vector of nonzero x
  arma::uvec nonzero_ind = arma::find(x != 0);
  arma::vec x_nonzero = x.elem(nonzero_ind);
  
  // Prepare output list
  List out;
  
  // Adjust the values
  arma::vec x2 = pow(x_nonzero, 2);
  arma::vec x2_df = x2 + (df - 2);
  arma::vec x_adj = (df - 2) / x2_df;
  
  // Indexes for positive elements
  arma::uvec pos_ind = arma::find(x_nonzero >= 0);
  arma::uvec neg_ind = arma::find(x_nonzero < 0);
  
  // Estimate probabilities
  NumericVector x_adj_num = wrap(x_adj);
  NumericVector prob_num = pbeta(x_adj_num, df / 2, 0.5);
  arma::vec prob = as<arma::vec>(prob_num);
  prob = 0.5 * prob;
  prob.elem(pos_ind) = 1 - prob.elem(pos_ind);
  
  // Calculate derivatives
  List diff;
  if (grad_x || grad_df)
  {
    diff = pbetaDiff(x_adj, df / 2, 0.5, n, false, R_NilValue);
  }
  
  // Values common for both derivatives
  arma::vec x2_adj;
  if (grad_x || grad_df)
  {
    x2_adj = pow(x2_df , 2);
  }
  
  // Derivative respect to x
  arma::vec grad_x_val;
  if (grad_x)
  {
    List dt0_list = dt0(x, df, false, false, false);
    arma::vec dt0_list_vec = dt0_list["den"];
    grad_x_val = dt0_list_vec;
  }
  
  // Derivative respect to df
  arma::vec grad_df_val;
  if (grad_df)
  {
    arma::vec dx = diff["dx"];
    arma::vec dp = diff["dp"];
    arma::vec dxdf = x2 / x2_adj;
    grad_df_val =  0.5 * (dx % dxdf + dp * 0.5);
    grad_df_val.elem(pos_ind) = -grad_df_val.elem(pos_ind);
    grad_df_val.elem(arma::find_nan(grad_df_val)).fill(0);
  }
  
  // Accurate routine for zero elements of x
  if (n_zero > 0)
  {
    // The number of observations
    const int n_obs = x.size();
    
    // Probabilities
    arma::vec prob_total(n_obs);
    prob_total.elem(zero_ind).fill(0.5);
    prob_total.elem(nonzero_ind) = prob;
    prob = prob_total;
    
    // Derivative respect to df
    if (grad_df)
    {
      arma::vec grad_df_total(n_obs);
      grad_df_total.elem(zero_ind).fill(0);
      grad_df_total.elem(nonzero_ind) = grad_df_val;
      grad_df_val = grad_df_total;
    }
  }
  
  // Deal with the logarithm if need
  if (log)
  {
    if (grad_x)
    {
      grad_x_val = grad_x_val / prob;
    }
    if (grad_df)
    {
      grad_df_val = grad_df_val / prob;
    }
    prob = arma::log(prob);
  }
  
  // Aggregate the output
  out["prob"] = prob;
  if (grad_x)
  {
    out["grad_x"] = grad_x_val;
  }
  if (grad_df)
  {
    out["grad_df"] = grad_df_val;
  }
  
  // Return the results
  return(out);
}

//' @name stdt
//' @export
// [[Rcpp::export(rng = true)]]
NumericVector rt0(const int n = 1,
                  const double df = 10)
{
  // Validation
  if (df <= 2)
  {
    stop ("Argument 'df' should be greater than 2.");
  }
  if (n < 1)
  {
    stop ("Argument 'n' should be a positive integer");
  }
  
  // Generate random variates
  NumericVector x = rt(n, df);
  x = x / sqrt(df / (df - 2));
  
  // Return the results
  return (x);
}

//' @name stdt
//' @export
// [[Rcpp::export(rng = true)]]
NumericVector qt0(const NumericVector x = 1,
                  const double df = 10)
{
  // Validation
  if (df <= 2)
  {
    stop ("Argument 'df' should be greater than 2.");
  }
   
  // Calculate the quantiles
  NumericVector q = qt(x, df, true, false);
  q = q / sqrt(df / (df - 2));
   
  // Return the results
  return (q);
}
