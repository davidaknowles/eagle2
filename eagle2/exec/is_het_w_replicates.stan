// Straightforward reimplementation of EAGLE using a beta-binomial likelihood. 
data {
  int<lower=0> N; 
  int<lower=0> T; 
  int<lower=0> R; 
  int<lower=0> ys[N,T,R];
  int<lower=0> ns[N,T,R];
  real<lower=0> concShape; 
  real<lower=0> concRate;  
  real<lower=0> errorRate; 
}
parameters {
  real<lower=0> conc; 
}
transformed parameters {
  vector[3] probs[N]; 
  for (n in 1:N) {
    probs[n] = rep_vector(0, 3);
    for (t in 1:T) {
      for (r in 1:R) {
        // likelihood of being het
        probs[n][1] = probs[n][1] + beta_binomial_lpmf(ys[n,t,r] | ns[n,t,r], conc * .5, conc * .5);
        // likelihood of being homozygous ref
        probs[n][2] = probs[n][2] + binomial_lpmf(ys[n,t,r] | ns[n,t,r], errorRate);
        // likelihood of being homozygous alt
        probs[n][3] = probs[n][3] + binomial_lpmf(ys[n,t,r] | ns[n,t,r], 1.0-errorRate);
      }
    }
  }
}
model {
  for (n in 1:N) {
    target += log_sum_exp(probs[n]);
  }
  conc ~ gamma(concShape, concRate);
}
