// EAGLE extension for repeated measurements of the same individual, in T conditions. 
// BB(n,a,sigma(phi*beta*x + r)) is the model, where phi in (-1,1) corresponding to the phase. 
// Regression coefficents are unconstrained. 
// Random effect terms r per SNP.
functions {
  // would be more efficient to pre-calc p
  real beta_binomial_reparam_lpmf(int y, int n, real g, real conc) {
    real p; 
    p = inv_logit(g);
    return beta_binomial_lpmf(y | n, conc*p, conc*(1.0-p));
  }
}
data {
  int<lower=0> N; // individuals
  int<lower=0> P; // covariates
  int<lower=0> K; // SNPs
  int<lower=0> T; // time points
  matrix[N,P] x[T]; 
  int<lower=0> ys[N,T,K];
  int<lower=0> ns[N,T,K];
  real<lower=0> concShape; 
  real<lower=0> concRate;
}
parameters {
  real<lower=0> conc[K];
  vector[P] beta;
  real<lower=0,upper=1> p[K]; 
  // TODO: should this be per gene? I guess not to deal with phase. 
  matrix[N,K] re; // random_effects 
  real log_rev; 
}
model {
  vector[N] xb[T]; 
  real rev; // random effect variance
  rev = exp(log_rev);
  to_vector(re) ~ normal(0,rev);
  for (t in 1:T)
    xb[t] = x[t] * beta;
  for (n in 1:N) {
    for (k in 1:K) {
      vector[T] log_prob_unflipped;
      vector[T] log_prob_flipped; 
      for (t in 1:T) {
        real g; 
        g = xb[t][n] + re[n,k];
        log_prob_unflipped[t] = beta_binomial_reparam_lpmf(ys[n,t,k] | ns[n,t,k], g, conc[k]);
        log_prob_flipped[t] = beta_binomial_reparam_lpmf(ys[n,t,k] | ns[n,t,k], -g, conc[k]);
      }
      target += log_sum_exp( log(p[k]) + sum(log_prob_unflipped), log(1.0-p[k]) + sum(log_prob_flipped) ) ;
    }
  }
  conc ~ gamma(concShape, concRate);
}
