% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{cmnorm}
\alias{cmnorm}
\title{Parameters of conditional multivariate normal distribution}
\usage{
cmnorm(
  mean,
  sigma,
  given_ind,
  given_x,
  dependent_ind = numeric(),
  is_validation = TRUE,
  is_names = TRUE,
  control = NULL,
  n_cores = 1L
)
}
\arguments{
\item{mean}{numeric vector representing expectation of multivariate
normal vector (distribution).}

\item{sigma}{positively defined numeric matrix representing covariance
matrix of multivariate normal vector (distribution).}

\item{given_ind}{numeric vector representing indexes of multivariate
normal vector which are conditioned at values given by 
\code{given_x} argument.}

\item{given_x}{numeric vector which \code{i}-th element corresponds to
the given value of the \code{given_ind[i]}-th element (component) of 
multivariate normal vector. If \code{given_x} is numeric matrix then it's 
rows are such vectors of given values.}

\item{dependent_ind}{numeric vector representing indexes of unconditional
elements (components) of multivariate normal vector.}

\item{is_validation}{logical value indicating whether input 
arguments should be validated.  Set it to \code{FALSE} to get
performance boost (default value is \code{TRUE}).}

\item{is_names}{logical value indicating whether output 
values should have row and column names. Set it to \code{FALSE} to get
performance boost (default value is \code{TRUE}).}

\item{control}{a list of control parameters. See Details.}

\item{n_cores}{positive integer representing the number of CPU cores
used for parallel computing. Currently it is not recommended to set
\code{n_cores > 1} if vectorized arguments include less then 100000 elements.}
}
\value{
This function returns an object of class "mnorm_cmnorm".\cr \cr
An object of class "mnorm_cmnorm" is a list containing the 
following components:
\itemize{
\item \code{mean} - conditional mean.
\item \code{sigma} - conditional covariance matrix.
\item \code{sigma_d} - covariance matrix of unconditioned elements.
\item \code{sigma_g} - covariance matrix of conditioned elements.
\item \code{sigma_dg} - matrix of covariances between unconditioned
and conditioned elements.
\item \code{s12s22} - equals to the matrix product of \code{sigma_dg}
and \code{solve(sigma_g)}.
}

Note that \code{mean} corresponds to \eqn{\mu_{c}} while \code{sigma}
represents \eqn{\Sigma_{c}}. Moreover \code{sigma_d} is 
\eqn{\Sigma_{I_{d}, I_{d}}}, \code{sigma_g} is \eqn{\Sigma_{I_{g}, I_{g}}} 
and \code{sigma_dg} is \eqn{\Sigma_{I_{d}, I_{g}}}.

Since \eqn{\Sigma_{c}} do not depend on
\eqn{X^{(g)}} the output \code{sigma} does not depend on \code{given_x}.
In particular output \code{sigma} remains the same independent of whether 
\code{given_x} is a matrix or vector. Oppositely if \code{given_x} is
a matrix then output \code{mean} is a matrix which rows correspond
to conditional means associated with given values provided by corresponding
rows of \code{given_x}.

The order of elements of output \code{mean} and output \code{sigma} depends 
on the order of \code{dependet_ind} elements that is ascending by default.
The order of \code{given_ind} elements does not matter. But, please, check 
that the order of \code{given_ind} match the order of given values i.e. 
the order of \code{given_x} columns.
}
\description{
This function calculates mean (expectation) and covariance 
matrix of conditional multivariate normal distribution.
}
\details{
Consider \eqn{m}-dimensional multivariate normal vector 
\eqn{X=(X_{1},...,X_{m})^{T}~\sim N(\mu,\Sigma)}, where \eqn{E(X)=\mu} and 
\eqn{Cov(X)=\Sigma} are expectation (mean) and covariance matrix 
respectively.

Let's denote vectors of indexes of conditioned and unconditioned elements of \eqn{X}
by \eqn{I_{g}} and \eqn{I_{d}} respectively. By \eqn{x^{(g)}} denote 
deterministic (column) vector of given values of \eqn{X_{I_{g}}}. The 
function calculates expected value and covariance matrix of conditioned
multivariate normal vector \eqn{X_{I_{d}} | X_{I_{g}} = x^{(g)}}. For example
if \eqn{I_{g}=(1, 3)} and \eqn{x^{(g)}=(-1, 1)} then \eqn{I_{d}=(2, 4, 5)} 
so the function calculates:
\deqn{\mu_{c}=E\left(\left(X_{2}, X_{4}, X_{5}\right) | X_{1} = -1, X_{3} = 1\right)}
\deqn{\Sigma_{c}=Cov\left(\left(X_{2}, X_{4}, X_{5}\right) | 
      X_{1} = -1, X_{3} = 1\right)}
      
In general case:
\deqn{\mu_{c} = E\left(X_{I_{d}} | X_{I_{g}} = x^{(g)}\right) = 
                \mu_{I_{d}} + 
                \left(x^{(g)} - \mu_{I_{g}}\right)
                \left(\Sigma_{(I_{d}, I_{g})}
                      \Sigma_{(I_{g}, I_{g})}^{-1}\right)^{T}}
\deqn{\Sigma_{c} = Cov\left(X_{I_{d}} | X_{I_{g}} = x^{(g)}\right) = 
                   \Sigma_{(I_{d}, I_{d})} - 
                   \Sigma_{(I_{d}, I_{g})}
                   \Sigma_{(I_{g}, I_{g})}^{-1}
                   \Sigma_{(I_{g}, I_{d})}}
Note that \eqn{\Sigma_{(A, B)}}, where \eqn{A,B\in\{d, g\}}, 
is a submatrix of \eqn{\Sigma} generated by intersection of \eqn{I_{A}} 
rows and \eqn{I_{B}} columns of \eqn{\Sigma}. 

Below there is a correspondence between aforementioned theoretical
(mathematical) notations and function arguments:
\itemize{
\item \code{mean} - \eqn{\mu}.
\item \code{sigma} - \eqn{\Sigma}.
\item \code{given_ind} - \eqn{I_{g}}.
\item \code{given_x} - \eqn{x^{(g)}}.
\item \code{dependent_ind} - \eqn{I_{d}}}
Moreover \eqn{\Sigma_{(I_{g}, I_{d})}} is a theoretical (mathematical)
notation for \code{sigma[given_ind, dependent_ind]}. Similarly \eqn{\mu_{g}}
represents \code{mean[given_ind]}.

By default \code{dependent_ind} contains all indexes that are not
in \code{given_ind}. It is possible to omit and duplicate indexes of 
\code{dependent_ind}. But at least single index should be provided for 
\code{given_ind} without any duplicates. Also \code{dependent_ind} and 
\code{given_ind} should not have the same elements. Moreover \code{given_ind}
should not be of the same length as \code{mean} so at least one component
should be unconditioned.

If \code{given_x} is a vector then (if possible) it will be treated as 
a matrix with the number of columns equal to the length of \code{mean}.

Currently \code{control} has no input arguments intended for
the users. This argument is used for some internal purposes
of the package.
}
\examples{
# Consider multivariate normal vector:
# X = (X1, X2, X3, X4, X5) ~ N(mean, sigma)

# Prepare multivariate normal vector parameters
  # expected value
mean <- c(-2, -1, 0, 1, 2)
n_dim <- length(mean)
  # correlation matrix
cor <- c(   1,  0.1,  0.2,   0.3,  0.4,
          0.1,    1, -0.1,  -0.2, -0.3,
          0.2, -0.1,    1,   0.3,  0.2,
          0.3, -0.2,  0.3,     1, -0.05,
          0.4, -0.3,  0.2, -0.05,     1)
cor <- matrix(cor, ncol = n_dim, nrow = n_dim, byrow = TRUE)
  # covariance matrix
sd_mat <- diag(c(1, 1.5, 2, 2.5, 3))
sigma <- sd_mat \%*\% cor \%*\% t(sd_mat)

# Estimate parameters of conditional distribution i.e.
# when the first and the third components of X are conditioned:
# (X2, X4, X5 | X1 = -1, X3 = 1)
given_ind <- c(1, 3)
given_x <- c(-1, 1)
par <- cmnorm(mean = mean, sigma = sigma,
              given_ind = given_ind,
              given_x = given_x)
  # E(X2, X4, X5 | X1 = -1, X3 = 1)
par$mean
  # Cov(X2, X4, X5 | X1 = -1, X3 = 1)
par$sigma

# Additionally calculate E(X2, X4, X5 | X1 = 2, X3 = 3)
given_x_mat <- rbind(given_x, c(2, 3))
par1 <- cmnorm(mean = mean, sigma = sigma,
               given_ind = given_ind,
               given_x = given_x_mat)
par1$mean

# Duplicates and omitted indexes are allowed for dependent_ind
# For given_ind duplicates are not allowed
# Let's calculate conditional parameters for (X5, X2, X5 | X1 = -1, X3 = 1):
dependent_ind <- c(5, 2, 5)
par2 <- cmnorm(mean = mean, sigma = sigma,
               given_ind = given_ind,
               given_x = given_x,
               dependent_ind = dependent_ind)
  # E(X5, X2, X5 | X1 = -1, X3 = 1)
par2$mean
  # Cov(X5, X2, X5 | X1 = -1, X3 = 1)
par2$sigma
}
