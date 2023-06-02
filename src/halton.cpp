#define ARMA_DONT_USE_OPENMP
#include <RcppArmadillo.h>
#include "halton.h"
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

#ifdef _OPENMP
// [[Rcpp::plugins(openmp)]]
#endif
// [[Rcpp::interfaces(r, cpp)]]

// Tasks
// 1. Add shuffling for 'halton'
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
//' @param scrambler string representing scrambling method for the 
//' Halton sequence. Possible options are \code{"NO"} (default), \code{"root"}
//' and \code{"negroot"} which described in S. Kolenikov (2012).
//' @template param_is_validation_Template
//' @template param_n_cores_Template
//' @details Function \code{\link[mnorm]{seqPrimes}} could be used to
//' provide the prime numbers for the \code{base} input argument.
//' @return The function returns a matrix which \code{i}-th column
//' is a sequence with base \code{base[i]} and elements with indexes
//' from \code{start} to \code{start + n}.
//' @references J. Halton (1964) <doi:10.2307/2347972>
//' @references S. Kolenikov (2012) <doi:10.1177/1536867X1201200103>
//' @examples halton(n = 100, base = c(2, 3, 5), start = 10)
// [[Rcpp::export(rng = true)]]
NumericMatrix halton(const int n = 1, 
                     const IntegerVector base = IntegerVector::create(2), 
                     const int start = 1,
                     const String random = "NO",
                     const String type = "halton",
                     const String scrambler = "NO",
                     const bool is_validation = true,
                     const int n_cores = 1)
{
  if (is_validation)
  {
    // Check that base are correctly specified
    if (any(base <= 1).is_true())
    {
      stop("Please, insure that all values in 'base' are greater than one.");
    }
    
    // Check that base are correctly specified
    if ((type != "halton") && (type != "richtmyer"))
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
  
  // Calculate elements of the Richtmyer sequence
  if (type == "richtmyer")
  {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if (n_cores > 1)
    #endif
    for(int b = 0; b < dim; b++)
    {
      double b_sqrt = std::sqrt(base[b]);
      for(int i = 0; i < n; i++)
      {
        h(i, b) = std::fmod((i + start) * b_sqrt, 1);
      }
    }
  }
  
  // Calculate elements of the Halton sequence
  // without scrambling
  if ((type == "halton"))
  {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_cores) if (n_cores > 1)
    #endif
    for(int b = 0; b < dim; b++)
    {
      for(int i = 0; i < n; i++)
      {
        h(i, b) = haltonSingleDraw(i + start, base[b], scrambler);
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
double haltonSingleDraw(int ind = 1, int base = 2, 
                        const String scrambler = "NO")
{
  double f = 1;
  double r = 0;
  
  if (scrambler == "NO")
  {
    while (ind > 0)
    {
      f = f / base;
      r = r + f * (ind % base);
      ind = ind / base;
    }
    return(r);
  }
  
  if (scrambler == "root")
  {
    int base_common = floor(sqrt(base));
    while (ind > 0)
    {
      f = f / base;
      r = r + f * ((base_common * (ind % base)) % base);
      ind = ind / base;
    }
    return(r);
  }
  
  if (scrambler == "negroot")
  {
    int base_common = base - (int)round(sqrt(base));
    while (ind > 0)
    {
      f = f / base;
      r = r + f * ((base_common * (ind % base)) % base);
      ind = ind / base;
    }
    return(r);
  }
  
  return(r);
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

//' Convert integer value to other base
//' @description Converts integer value to other base.
//' @param x positive integer representing the number to convert.
//' @param base positive integer representing the base.
//' @return The function returns a numeric vector containing 
//' representation of \code{x} in a base given in \code{base}.
//' @examples toBase(888, 5)
// [[Rcpp::export(rng = false)]]
IntegerVector toBase(int x, const int base = 2)
{
  IntegerVector val;
  
  // Apply Horner's method
  while (x > 0)
  {
    val.push_front(x % base);
    x = x / base;
  }
  
  return(val);
}

//' Convert base representation of a number into integer
//' @description Converts base representation of a number into integer.
//' @param x vector of positive integer coefficients representing the number
//' in base that is \code{base}.
//' @param base positive integer representing the base.
//' @return The function returns a positive integer that is a
//' conversion from \code{base} under given coefficients \code{x}.
//' @examples fromBase(c(1, 2, 0, 2, 3), 5)
// [[Rcpp::export(rng = false)]]
double fromBase(const IntegerVector x, const int base = 2)
{
  const int n = x.size();

  int mult = 1;
  int y = 0;
  for (int i = n - 1; i >= 0; i--)
  {
    y += x(i) * mult;
    mult *= base;
  }
  
  return(y);
}
