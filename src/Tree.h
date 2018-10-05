#ifndef SIMPLERANDOMFOREST_TREE_H

#include <RcppArmadillo.h>
#include <vector>
#include <random>
#include <utility>

class Tree {
public:
  Tree(
    const arma::mat &x, const arma::uvec &y,
    unsigned int mtry,
    bool replace, double sample_fraction,
    unsigned int seed
  );
  void grow();
  void init(
      const std::vector<size_t> &split_var,
      const std::vector<double> &split_value,
      const std::vector<size_t> &split_child_left, const std::vector<size_t> &split_child_right
  );
  arma::uword predict(size_t smp) const;
  
  const std::vector<std::vector<size_t>> &getSamples() const { return samples; }
  const std::vector<size_t> &getSplitVariables() const { return split_var; }
  const std::vector<double> &getSplitValues() const { return split_value; }
  const std::vector<size_t> &getChildIndicesLeft() const { return split_child_left; }
  const std::vector<size_t> &getChildIndicesRight() const { return split_child_right; }
  const arma::vec &getImportance() const { return importance; }

private:
  const arma::mat &x;
  const arma::uvec &y;
  const unsigned int mtry;
  const bool replace;
  const double sample_fraction;
  
  arma::uvec y_levels;
  std::vector<size_t> inbag;
  std::mt19937_64 rand;
  
  std::vector<std::vector<size_t>> samples;
  std::vector<size_t> split_var;
  std::vector<double> split_value;
  std::vector<size_t> split_child_left, split_child_right;
  
  arma::vec importance;
  
  void bootstrap();
  void splitNode(size_t split_index);
  size_t createNode();
  void addGiniImportance(size_t split_index, size_t var, double decrease);
};

#endif
