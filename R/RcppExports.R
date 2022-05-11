# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#' Parameters of conditional multivariate normal distribution
#' @description This function calculates mean (expectation) and covariance 
#' matrix of conditional multivariate normal distribution.
#' @template param_mean_Template
#' @template param_sigma_Template
#' @template param_given_ind_Template
#' @template param_given_x_Template
#' @template param_dependent_ind_Template
#' @template param_is_validation_Template
#' @template param_is_names_Template
#' @template param_control_Template
#' @template param_n_cores_Template
#' @template details_cmnorm_Template
#' @template return_cmnorm_Template
#' @template example_cmnorm_Template
#' @export
cmnorm <- function(mean, sigma, given_ind, given_x, dependent_ind = numeric(), is_validation = TRUE, is_names = TRUE, control = NULL, n_cores = 1L) {
    .Call(`_mnorm_cmnorm`, mean, sigma, given_ind, given_x, dependent_ind, is_validation, is_names, control, n_cores)
}

#' Density of (conditional) multivariate normal distribution
#' @description This function calculates and differentiates density of 
#' (conditional) multivariate normal distribution.
#' @template param_x_Template
#' @template param_mean_Template
#' @template param_sigma_Template
#' @template param_given_ind_2_Template
#' @template param_log_Template
#' @template param_grad_x_Template
#' @template param_grad_sigma_Template
#' @template param_is_validation_Template
#' @template param_control_Template
#' @template param_n_cores_Template
#' @template details_dmnorm_Template
#' @template return_dmnorm_Template
#' @template example_dmnorm_Template
#' @references E. Kossova., B. Potanin (2018). 
#' Heckman method and switching regression model multivariate generalization.
#' Applied Econometrics, vol. 50, pages 114-143.
#' @export
dmnorm <- function(x, mean, sigma, given_ind = numeric(), log = FALSE, grad_x = FALSE, grad_sigma = FALSE, is_validation = TRUE, control = NULL, n_cores = 1L) {
    .Call(`_mnorm_dmnorm`, x, mean, sigma, given_ind, log, grad_x, grad_sigma, is_validation, control, n_cores)
}

#' Halton sequence
#' @description Calculate elements of the Halton sequence and of
#' some other pseudo-random sequences.
#' @param n positive integer representing the number of sequence elements.
#' @param base vector of positive integers greater then one representing
#' the bases for each of the sequences.
#' @param start non-negative integer representing the index of the first 
#' element of the sequence to be included in the output sequence.
#' @param random string representing the method of randomization to be
#' applied to the sequence. If \code{random = "NO"} (default) then
#' there is no randomization. If \code{random = "Tuffin"} then standard uniform
#' random variable will be added to each element of the sequence and 
#' the difference between this sum and it's 'floor' will be returned as
#' a new element of the sequence.
#' @param type string representing type of the sequence. Default is "halton"
#' that is Halton sequence. The alternative is "richtmyer" corresponding 
#' to Richtmyer sequence.
#' @template param_is_validation_Template
#' @template param_n_cores_Template
#' @details Function \code{\link[mnorm]{seqPrimes}} could be used to
#' provide the prime numbers for the \code{base} input argument.
#' @return The function returns a matrix which \code{i}-th column
#' is a sequence with base \code{base[i]} and elements with indexes
#' from \code{start} to \code{start + n}.
#' @references J. Halton (1964) <doi:10.2307/2347972>
#' @examples halton(n = 100, base = c(2, 3, 5), start = 10)
halton <- function(n = 1L, base = as.integer( c(2)), start = 1L, random = "NO", type = "halton", is_validation = TRUE, n_cores = 1L) {
    .Call(`_mnorm_halton`, n, base, start, random, type, is_validation, n_cores)
}

haltonSingleDraw <- function(ind = 1L, base = 2L) {
    .Call(`_mnorm_haltonSingleDraw`, ind, base)
}

#' Sequence of prime numbers
#' @description Calculates the sequence of prime numbers.
#' @param n positive integer representing the number of sequence elements.
#' @return The function returns a numeric vector containing 
#' first \code{n} prime numbers. The current (naive) implementation of the 
#' algorithm is not efficient in terms of speed so it is suited for low 
#' \code{n < 10000} but requires just O(n) memory usage.
#' @examples seqPrimes(10)
seqPrimes <- function(n) {
    .Call(`_mnorm_seqPrimes`, n)
}

#' Probabilities of (conditional) multivariate normal distribution
#' @description This function calculates and differentiates probabilities of
#' (conditional) multivariate normal distribution.
#' @template details_pmnorm_Template
#' @template param_lower_Template
#' @template param_upper_Template
#' @template param_given_x_Template
#' @template param_mean_Template
#' @template param_sigma_Template
#' @template param_given_ind_Template
#' @template param_n_sim_Template
#' @template param_method_Template
#' @template param_ordering_Template
#' @template param_log_Template
#' @template param_grad_lower_Template
#' @template param_grad_upper_Template
#' @template param_grad_sigma_pmnorm_Template
#' @template param_grad_given_Template
#' @template param_is_validation_Template
#' @template param_control_Template
#' @template param_n_cores_Template
#' @template return_pmnorm_Template
#' @template example_pmnorm_Template
#' @references Genz, A. (2004), Numerical computation of rectangular bivariate 
#' and trivariate normal and t-probabilities, Statistics and 
#' Computing, 14, 251-260.
#' @references Genz, A. and Bretz, F. (2009), Computation of Multivariate 
#' Normal and t Probabilities. Lecture Notes in Statistics, Vol. 195. 
#' Springer-Verlag, Heidelberg.
#' @references E. Kossova., B. Potanin (2018). 
#' Heckman method and switching regression model multivariate generalization.
#' Applied Econometrics, vol. 50, pages 114-143.
#' @export
pmnorm <- function(lower, upper, given_x = numeric(), mean = numeric(), sigma = matrix(), given_ind = numeric(), n_sim = 1000L, method = "default", ordering = "mean", log = FALSE, grad_lower = FALSE, grad_upper = FALSE, grad_sigma = FALSE, grad_given = FALSE, is_validation = TRUE, control = NULL, n_cores = 1L) {
    .Call(`_mnorm_pmnorm`, lower, upper, given_x, mean, sigma, given_ind, n_sim, method, ordering, log, grad_lower, grad_upper, grad_sigma, grad_given, is_validation, control, n_cores)
}

pmnorm2 <- function(x1, x2, x, adj, adj1, adj2, n_cores = 1L) {
    .Call(`_mnorm_pmnorm2`, x1, x2, x, adj, adj1, adj2, n_cores)
}

GHK <- function(lower, upper, sigma, h, ordering = "default", n_sim = 1000L, n_cores = 1L) {
    .Call(`_mnorm_GHK`, lower, upper, sigma, h, ordering, n_sim, n_cores)
}

#' Quantile function of a normal distribution
#' @description Calculate quantile of a normal distribution using
#' one of the available methods.
#' @param p numeric vector of values between 0 and 1 representing levels of
#' the quantiles.
#' @param mean numeric value representing the expectation of a
#' normal distribution.
#' @param sd positive numeric value representing standard deviation of a
#' normal distribution.
#' @param method character representing the method to be used for
#' quantile calculation. Available options are "Voutier" (default) and "Shore".
#' @template param_is_validation_Template
#' @template param_n_cores_Template
#' @details If \code{method = "Voutier"} then the method of P. Voutier (2010)
#' is used which maximum absolute error is about \eqn{0.000025}.
#' If \code{method = "Shore"} then the approach proposed
#' by H. Shore (1982) is applied which maximum absolute error is about
#' \eqn{0.026} for quantiles of level between \eqn{0.0001} 
#' and \eqn{0.9999}.
#' @return The function returns a vector of \code{p}-level quantiles of a
#' normal distribution with mean equal to \code{mean} and standard 
#' deviation equal to \code{sd}.
#' @references H. Shore (1982) <doi:10.2307/2347972>
#' @references P. Voutier (2010) <doi:10.48550/arXiv.1002.0567>
#' @examples qnormFast(c(0.1, 0.9), mean = 1, sd = 2)
qnormFast <- function(p, mean = 0L, sd = 1L, method = "Voutier", is_validation = TRUE, n_cores = 1L) {
    .Call(`_mnorm_qnormFast`, p, mean, sd, method, is_validation, n_cores)
}

# Register entry points for exported C++ functions
methods::setLoadAction(function(ns) {
    .Call('_mnorm_RcppExport_registerCCallable', PACKAGE = 'mnorm')
})