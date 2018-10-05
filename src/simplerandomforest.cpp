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
    size_t num_threads
) {
  if(num_threads == 0) num_threads = std::thread::hardware_concurrency();
  
  Forest forest(x, y, num_trees, mtry, replace, sample_fraction, num_threads);
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
    Named("importance") = forest.getImportance(),
    Named("oob.error") = forest.getOOBError()
  );
  
  return out;
}
