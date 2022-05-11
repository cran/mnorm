#include <RcppArmadillo.h>
#include "halton.h"
using namespace Rcpp;

// [[Rcpp::interfaces(r, cpp)]]

// Tasks
// 1. Add scrambling and shuffling for 'halton'
// 2. Add Sieve of Eratosthenes and Atkin for 'seqPrimes'

//' Halton sequence
//' @description Calculate elements of the Halton sequence and of
//' some other pseudo-random sequences.
//' @param n positive integer representing the number of sequence elements.
//' @param base vector of positive integers greater then one representing
//' the bases for each of the sequences.
//' @param start non-negative integer representing the index of the first 
//' element of the sequence to be included in the output sequence.
//' @param random string representing the method of randomization to be
//' applied to the sequence. If \code{random = "NO"} (default) then
//' there is no randomization. If \code{random = "Tuffin"} then standard uniform
//' random variable will be added to each element of the sequence and 
//' the difference between this sum and it's 'floor' will be returned as
//' a new element of the sequence.
//' @param type string representing type of the sequence. Default is "halton"
//' that is Halton sequence. The alternative is "richtmyer" corresponding 
//' to Richtmyer sequence.
//' @template param_is_validation_Template
//' @template param_n_cores_Template
//' @details Function \code{\link[mnorm]{seqPrimes}} could be used to
//' provide the prime numbers for the \code{base} input argument.
//' @return The function returns a matrix which \code{i}-th column
//' is a sequence with base \code{base[i]} and elements with indexes
//' from \code{start} to \code{start + n}.
//' @references J. Halton (1964) <doi:10.2307/2347972>
//' @examples halton(n = 100, base = c(2, 3, 5), start = 10)
// [[Rcpp::export(rng = true)]]
NumericMatrix halton(const int n = 1, 
                     const IntegerVector base = IntegerVector::create(2), 
                     const int start = 1,
                     const String random = "NO",
                     const String type = "halton",
                     const bool is_validation = true,
                     const int n_cores = 1)
{
  // Multiple cores
  omp_set_num_threads(n_cores);
  
  
  if (is_validation)
  {
    // Check that base are correctly specified
    if (any(base <= 1).is_true())
    {
      stop("Please, insure that all values in 'base' are greater than one.");
    }
    
    // Check that base are correctly specified
    if ((type != "halton") & (type != "richtmyer"))
    {
      stop("Please, insure that 'type' has been provided with a correct value.");
    }
    
    // Check that the number of cores is correctly specified
    if (n_cores < 1)
    {
      stop("Please, insure that 'n_cores' is a positive integer.");
    }
  
    // Check that start argument is correct
    if (start <= 0)
    {
      stop("Please, insure that 'start' is a positive integer.");
    }
  }
  
  // Get the number of Halton sequences
  const int dim = base.size();

  // Matrix to store Halton sequence for each base
  NumericMatrix h = NumericMatrix(n, dim);
  
  // Calculate elements of the richtmyer sequence
  if (type == "richtmyer")
  {
    for(int b = 0; b < dim; b++)
    {
      double b_sqrt = sqrt(base[b]);
      for(int i = 0; i < n; i++)
      {
        h(i, b) = std::fmod((i + start) * b_sqrt, 1);
      }
    }
  }
  
  // Calculate elements of the Halton sequence
  if (type == "halton")
  {
    for(int b = 0; b < dim; b++)
    {
      for(int i = 0; i < n; i++)
      {
        h(i, b) = haltonSingleDraw(i + start, base[b]);
      }
    }
  }
  
  // Randomization
  if (random == "Tuffin")
  {
    for(int b = 0; b < dim; b++)
    {
      NumericVector u = runif(n, 0.0, 1.0);
      NumericVector hb = h(_, b);
      hb = hb + u;
      hb = hb - floor(hb);
      h(_, b) = hb;
    }
  }
  
  return(h);
}

// [[Rcpp::export(rng = false)]]
double haltonSingleDraw(int ind = 1, int base = 2)
{
  double f = 1;
  double r = 0;
  
  while (ind > 0)
  {
    f = f / base;
    r = r + f * (ind % base);
    ind = ind / base;
  }
  
  return r;
}

//' Sequence of prime numbers
//' @description Calculates the sequence of prime numbers.
//' @param n positive integer representing the number of sequence elements.
//' @return The function returns a numeric vector containing 
//' first \code{n} prime numbers. The current (naive) implementation of the 
//' algorithm is not efficient in terms of speed so it is suited for low 
//' \code{n < 10000} but requires just O(n) memory usage.
//' @examples seqPrimes(10)
// [[Rcpp::export(rng = false)]]
IntegerVector seqPrimes(const int n)
{
  // Validation
  if (n <= 0)
  {
    stop("Please, insure that 'n' is a positive integer.");
  }
  
  // Routine
  IntegerVector primes(n);
  primes[0] = 2;
  
  int n_find = 1;
  int i = 2;
  
  while (n_find < n) 
  {
    i++;
    bool prime = true;
    for (int j = 0; j < n_find; j++)
    {
      if (i % primes[j] == 0) 
      {
        prime = false;
        break;    
      }
    }   
    if(prime)
    {
      primes[n_find] = i;
      n_find++;
    }
  }
  
  return(primes);
}
