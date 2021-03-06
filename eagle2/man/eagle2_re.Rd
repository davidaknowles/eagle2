% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/eagle2_re.R
\name{eagle2_re}
\alias{eagle2_re}
\title{Beta binomial GLMM with flips. Prior on concentration parameter is Gamma(concShape,concRate)}
\usage{
eagle2_re(ys, ns, concShape = 1.0001, concRate = 1e-04,
  USE_LBFGS = T, burnin = 3000, iterations = 1000,
  elbo_samples = 1000, learning_rate = 1, seed = 1, ...)
}
\arguments{
\item{ys}{numerator counts [n x T x K] where n are individuals, T are timepoints, K are SNPs}

\item{ns}{denominator counts [n x T x K]}

\item{concShape}{Shape of prior on concentration}

\item{concRate}{Rate of prior on concentration}
}
\value{
List with likelihood ratio, p-value and fits
}
\description{
Includes a per individual, per SNP random effect (shared across conditions) and uses stochastic 
variational inference to integrate over these.
}
