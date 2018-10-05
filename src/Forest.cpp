#include "Forest.h"

#include <random>
#include <thread>
#include <algorithm>

Forest::Forest(const arma::mat &x, const arma::uvec &y, unsigned int num_trees, unsigned int mtry, bool replace, double sample_fraction, unsigned int num_threads) :
  x(x), y(y), num_trees(num_trees), mtry(mtry), replace(replace), sample_fraction(sample_fraction), num_threads(num_threads)
{
  std::random_device rd;
  std::mt19937_64 rand(rd());
  std::uniform_int_distribution<size_t> udist;
  for(size_t i = 0; i < num_trees; ++i) {
    trees.emplace_back(x, y, mtry, replace, sample_fraction, udist(rand));
  }
}

void Forest::grow() {
  std::vector<std::thread> threads;
  for(size_t tid = 0; tid < std::min(num_threads, num_trees); ++tid) {
    threads.push_back(std::thread([&](size_t offset) {
      for(size_t i = offset; i < num_trees; i += num_threads) {
        trees[i].grow();
      }
    }, tid));
  }
  
  for(std::thread &th : threads) th.join();
}

void Forest::init(const Rcpp::List &trees) {
  for(size_t i = 0; i < num_trees; ++i) {
    const Rcpp::List &tree = trees[i];
    std::vector<size_t> split_var = tree["split_var"];
    std::vector<double> split_value = tree["split_value"];
    std::vector<size_t> split_child_left = tree["split_child_left"];
    std::vector<size_t> split_child_right = tree["split_child_right"];
    this->trees[i].init(split_var, split_value, split_child_left, split_child_right);
  }
}

arma::uvec Forest::predict() const {
  arma::uvec out(x.n_rows);
  for(size_t i = 0; i < x.n_rows; ++i) {
    arma::uvec pred(num_trees);
    for(size_t j = 0; j < num_trees; ++j) {
      pred[j] = trees[j].predict(i);
    }
    
    arma::uword y_max = arma::max(pred);
    arma::uvec sp = arma::linspace<arma::uvec>(0, y_max, y_max+1);
    arma::uvec y_hist = arma::hist(pred, sp);
    out[i] = arma::index_max(y_hist);
  }
  return out;
}

const arma::vec Forest::getImportance() const {
  arma::vec imp(x.n_cols, arma::fill::zeros);
  for(const Tree &tree : trees) {
    imp += tree.getImportance();
  }
  return imp / trees.size();
}
