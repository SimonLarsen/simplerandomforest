#include <RcppArmadillo.h>
#include <thread>
#include "Forest.h"

using namespace Rcpp;

// [[Rcpp::export]]
List simplerandomforestCpp(
    const arma::mat &x,
    const arma::uvec &y,
    size_t num_trees,
    size_t mtry,
    bool replace,
    double sample_fraction,
    bool permutation_importance,
    size_t num_threads
) {
  if(num_threads == 0) num_threads = std::thread::hardware_concurrency();
  
  arma::uvec y_levels = arma::unique(y);
  Forest forest(x, y, y_levels.n_elem, num_trees, mtry, replace, sample_fraction, num_threads);
  forest.grow();
  
  List out_trees;
  for(const Tree &tree : forest.getTrees()) {
    List out_tree = List::create(
      Named("samples") = tree.getSamples(),
      Named("split_var") = tree.getSplitVariables(),
      Named("split_value") = tree.getSplitValues(),
      Named("split_child_left") = tree.getChildIndicesLeft(),
      Named("split_child_right") = tree.getChildIndicesRight()
    );
    out_trees.push_back(out_tree);
  }
  
  List out = List::create(
    Named("trees") = out_trees,
    Named("gini.importance") = forest.getGiniImportance(),
    Named("oob.error") = forest.getOOBError()
  );
  if(permutation_importance) out["permutation.importance"] = forest.getPermutationImportance();
  
  return out;
}
