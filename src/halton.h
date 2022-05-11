#ifndef mnorm_halton_H
#define mnorm_halton_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace RcppArmadillo;

NumericMatrix halton(const int n, 
                     const IntegerVector base, 
                     const int start,
                     const String random,
                     const String type,
                     const bool is_validation,
                     const int n_cores);

double haltonSingleDraw(int ind, int base);

IntegerVector seqPrimes(const int n);

#endif
