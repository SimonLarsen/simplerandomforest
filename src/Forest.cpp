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
  
  computeOOBError();
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
  arma::uvec y_levels = arma::unique(y);
  arma::uvec y_space = arma::linspace<arma::uvec>(0, y_levels.n_elem-1, y_levels.n_elem);
  
  for(size_t i = 0; i < x.n_rows; ++i) {
    arma::uvec counts(y_levels.n_elem, arma::fill::zeros);
    for(const Tree &tree : trees) {
      size_t pred = tree.predict(i);
      counts[pred]++;
    }
    out[i] = arma::index_max(counts);
  }
  return out;
}

void Forest::computeOOBError() {
  arma::uvec y_levels = arma::unique(y);
  arma::umat counts(x.n_rows, y_levels.n_elem, arma::fill::zeros);
  arma::uvec y_space = arma::linspace<arma::uvec>(0, y_levels.n_elem-1, y_levels.n_elem);
  
  for(const Tree &tree : trees) {
    const std::vector<size_t> outofbag = tree.getOOBSamples();
    for(size_t i : outofbag) {
      size_t pred = tree.predict(i);
      counts(i, pred) += 1;
    }
  }
  
  int errors = 0;
  for(size_t i = 0; i < x.n_rows; ++i) {
    size_t pred = arma::index_max(counts.row(i));
    if(pred != y(i)) errors++;
  }
  
  oob_error = (double)errors / x.n_rows;
}

const arma::vec Forest::getImportance() const {
  arma::vec imp(x.n_cols, arma::fill::zeros);
  for(const Tree &tree : trees) {
    imp += tree.getImportance();
  }
  return imp / trees.size();
}

const double Forest::getOOBError() const {
  return oob_error;
}