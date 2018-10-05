#ifndef SIMPLERANDOMFOREST_FOREST_H

#include <RcppArmadillo.h>
#include <vector>
#include "Tree.h"

class Forest {
public:
  Forest(
    const arma::mat &x, const arma::uvec &y,
    unsigned int num_trees, unsigned int mtry,
    bool replace, double sample_fraction,
    unsigned int num_threads
  );
  void grow();
  void init(const Rcpp::List &trees);
  arma::uvec predict() const;
  const arma::vec getImportance() const;
  
  const std::vector<Tree> &getTrees() const { return trees; }
  
private:
  const arma::mat &x;
  const arma::uvec &y;
  const unsigned int num_trees, mtry;
  const bool replace;
  const double sample_fraction;
  const unsigned int num_threads;
  
  std::vector<Tree> trees;
};

#endif
