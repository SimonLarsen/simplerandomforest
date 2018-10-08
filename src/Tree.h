#ifndef SIMPLERANDOMFOREST_TREE_H
#define SIMPLERANDOMFOREST_TREE_H

#include <RcppArmadillo.h>
#include <vector>
#include <random>
#include <utility>

class Tree {
public:
  Tree(
    const arma::mat &x, const arma::uvec &y, unsigned int y_levels,
    unsigned int mtry,
    bool replace, double sample_fraction,
    std::uint32_t seed
  );
  void grow();
  void init(
      const std::vector<size_t> &split_var,
      const std::vector<double> &split_value,
      const std::vector<size_t> &split_child_left, const std::vector<size_t> &split_child_right
  );
  arma::uword predict(size_t smp) const;
  arma::vec computePermutationImportance();
  
  const std::vector<std::vector<size_t>> &getSamples() const { return samples; }
  const std::vector<size_t> &getOOBSamples() const { return outofbag; }
  const std::vector<size_t> &getSplitVariables() const { return split_var; }
  const std::vector<double> &getSplitValues() const { return split_value; }
  const std::vector<size_t> &getChildIndicesLeft() const { return split_child_left; }
  const std::vector<size_t> &getChildIndicesRight() const { return split_child_right; }
  const arma::vec &getGiniImportance() const { return gini_importance; }

private:
  const arma::mat &x;
  const arma::uvec &y;
  const unsigned int y_levels;
  const unsigned int mtry;
  const bool replace;
  const double sample_fraction;
  
  std::vector<size_t> inbag, outofbag;
  std::mt19937_64 rand;
  
  std::vector<std::vector<size_t>> samples;
  std::vector<size_t> split_var;
  std::vector<double> split_value;
  std::vector<size_t> split_child_left, split_child_right;
  
  arma::vec gini_importance;
  
  void bootstrap();
  void splitNode(size_t split_index);
  size_t createNode();
  void addGiniImportance(size_t split_index, size_t var, double decrease);
  double computeOOBError() const;
  arma::uword predictPermuted(size_t smp, size_t smp_permuted, size_t var);
};

#endif
