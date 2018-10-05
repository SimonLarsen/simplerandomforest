#include <RcppArmadillo.h>
#include <thread>
#include "Forest.h"

// [[Rcpp::export]]
arma::uvec predictCpp(
    const Rcpp::List &model,
    const arma::mat &x,
    size_t num_threads
) {
  if(num_threads == 0) num_threads = std::thread::hardware_concurrency();
  
  arma::uvec y;
  Forest forest(x, y, model["num.trees"], 0, false, 0, num_threads);
  forest.init(model["trees"]);
  
  return forest.predict();
}
