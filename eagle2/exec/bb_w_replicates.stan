// EAGLE extension for repeated measurements of the same individual, in T conditions. 
// BB(n,a,sigma(phi*beta*x)) is the model, where phi in (-1,1) corresponding to the phase. 
// This version allows for K SNPs
// No constraints on cofficients.
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
  int<lower=0> T; // time points`
  int<lower=0> R; // replicates
  matrix[N,P] x[T]; // covariates
  int<lower=0> ys[N,T,K,R]; // minor allele counts
  int<lower=0> ns[N,T,K,R]; // total coverage counts
  real<lower=0> concShape; // gamma prior on concentration parameters
  real<lower=0> concRate;
}
parameters {
  real<lower=0> conc[K]; // per exonic SNP concentration parameters
  vector[P] beta; // effect sizes
  real<lower=0,upper=1> p[K]; // learnt prior on phase (per SNP)
}
model {
  vector[N] xb[T]; 
  for (t in 1:T)
    xb[t] = x[t] * beta;
  for (n in 1:N) {
    for (k in 1:K) {
      vector[T] log_prob_unflipped;
      vector[T] log_prob_flipped; 
      for (t in 1:T) {
        vector[R] log_prob_unflipped_temp; 
        vector[R] log_prob_flipped_temp; 
        for (r in 1:R) {
          log_prob_unflipped_temp[r] = beta_binomial_reparam_lpmf(ys[n,t,k,r] | ns[n,t,k,r], xb[t][n], conc[k]); 
          log_prob_flipped_temp[r] = beta_binomial_reparam_lpmf(ys[n,t,k,r] | ns[n,t,k,r], -xb[t][n], conc[k]);
        }
        log_prob_unflipped[t]=sum(log_prob_unflipped_temp);
        log_prob_flipped[t]=sum(log_prob_flipped_temp);
      }
      target += log_sum_exp( log(p[k]) + sum(log_prob_unflipped), log(1.0-p[k]) + sum(log_prob_flipped) ) ;
    }
  }
  conc ~ gamma(concShape, concRate);
}
