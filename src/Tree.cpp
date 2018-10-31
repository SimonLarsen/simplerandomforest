#include "Tree.h"
#include <cmath>
#include <algorithm>
#include <array>

Tree::Tree(
  const arma::mat &x, const arma::uvec &y, unsigned int y_levels,
  unsigned int mtry,
  bool replace, double sample_fraction,
  std::uint32_t seed
) : x(x), y(y), y_levels(y_levels), mtry(mtry), replace(replace), sample_fraction(sample_fraction), rand(seed)
{ }

void Tree::init(
    const std::vector<size_t> &split_var,
    const std::vector<double> &split_value,
    const std::vector<size_t> &split_child_left,
    const std::vector<size_t> &split_child_right
) {
  this->split_var = split_var;
  this->split_value = split_value;
  this->split_child_left = split_child_left;
  this->split_child_right = split_child_right;
}

void Tree::grow() {
  bootstrap();
  
  createNode();
  std::copy(inbag.begin(), inbag.end(), std::back_inserter(samples[0]));
  
  gini_importance = arma::vec(x.n_cols, arma::fill::zeros);

  size_t i = 0;
  while(i < split_var.size()) {
    splitNode(i);
    ++i;
  }
}

arma::uword Tree::predict(size_t smp) const {
  size_t node = 0;
  while(true) {
    if(split_child_left[node] == 0) {
      return split_var[node];
    } else {
      if(x.at(smp, split_var[node]) < split_value[node]) {
        node = split_child_left[node];
      } else {
        node = split_child_right[node];
      }
    }
  }
}

void Tree::bootstrap() {
  size_t nsamples = (size_t)std::floor(x.n_rows * sample_fraction);
  if(replace) {
    std::uniform_int_distribution<size_t> udist(0, x.n_rows-1);
    for(size_t i = 0; i < nsamples; ++i) {
      inbag.push_back(udist(rand));
    }
  } else {
    inbag.resize(x.n_rows);
    std::iota(inbag.begin(), inbag.end(), 0);
    std::shuffle(inbag.begin(), inbag.end(), rand);
    inbag.resize(nsamples);
  }
  
  std::sort(inbag.begin(), inbag.end());
  std::vector<size_t> allsamples(x.n_rows);
  std::iota(allsamples.begin(), allsamples.end(), 0);
  outofbag.resize(inbag.size());
  std::vector<size_t>::iterator outofbag_end = std::set_difference(allsamples.begin(), allsamples.end(), inbag.begin(), inbag.end(), outofbag.begin());
  outofbag.resize(outofbag_end - outofbag.begin());
}

void Tree::splitNode(size_t split_index) {
  // Check if node is pure
  arma::uword first_response = y[samples[split_index][0]];
  bool pure = true;
  for(size_t smp : samples[split_index]) {
    if(y[smp] != first_response) {
      pure = false;
      break;
    }
  }
  
  if(pure) {
    split_var[split_index] = first_response;
    return;
  }
  
  // Sample candidate variables
  std::vector<size_t> candidate_vars(x.n_cols);
  std::iota(candidate_vars.begin(), candidate_vars.end(), 0);
  std::shuffle(candidate_vars.begin(), candidate_vars.end(), rand);
  candidate_vars.resize(mtry);
  
  size_t best_var;
  double best_value;
  double best_decrease = -1;
  
  // Evaluate candidate values and split 
  arma::mat assign(2, y_levels);
  for(size_t var : candidate_vars) {
    //arma::vec values = arma::unique(x.col(var));
    for(double value : x.col(var)) {
      assign.fill(0);
      for(size_t smp : samples[split_index]) {
        if(x.at(smp, var) < value) {
          assign.at(0, y[smp])++;
        } else {
          assign.at(1, y[smp])++;
        }
      }
      
      if(arma::sum(assign.row(0)) == 0 || arma::sum(assign.row(1)) == 0) continue;
      
      arma::vec rs = arma::sum(assign, 1);
      double decrease = arma::sum(arma::sum(arma::square(assign), 1) / rs);
      
      if(decrease > best_decrease) {
        best_decrease = decrease;
        best_var = var;
        best_value = value;
      }
    }
  }
  
  if(best_decrease <= 0) {
    std::vector<size_t> counts(y_levels, 0);
    for(size_t smp : samples[split_index]) {
      counts[y[smp]]++;
    }
    std::vector<size_t>::iterator max_it = std::max_element(counts.begin(), counts.end());
    split_var[split_index] = std::distance(counts.begin(), std::max_element(counts.begin(), counts.end()));
    return;
  }
  
  split_var[split_index] = best_var;
  split_value[split_index]  = best_value;
  
  size_t left_index = createNode();
  size_t right_index = createNode();
  split_child_left[split_index] = left_index;
  split_child_right[split_index] = right_index;
  
  for(size_t smp : samples[split_index]) {
    if(x.at(smp, best_var) < best_value) {
      samples[left_index].push_back(smp);
    } else {
      samples[right_index].push_back(smp);
    }
  }
  
  addGiniImportance(split_index, best_var, best_decrease);
}

size_t Tree::createNode() {
  size_t index = samples.size();
  
  samples.emplace_back();
  split_var.push_back(0);
  split_value.push_back(0);
  split_child_left.push_back(0);
  split_child_right.push_back(0);
  
  return index;
}

void Tree::addGiniImportance(size_t split_index, size_t var, double decrease) {
  std::vector<int> counts(y_levels, 0);
  for(size_t smp : samples[split_index]) {
    arma::uword value = y[smp];
    counts[value]++;
  }
  double sum = 0;
  for(int count : counts) {
    sum += count * count;
  }
  
  double node_decrease = decrease - sum / (double)samples[split_index].size();
  gini_importance[var] += node_decrease;
}

double Tree::computeOOBError() const {
  int errors = 0;
  for(size_t smp : outofbag) {
    size_t pred = predict(smp);
    if(pred != y[smp]) errors++;
  }
  return (double)errors / outofbag.size();
}

arma::vec Tree::computePermutationImportance() {
  double base_error = computeOOBError();
  
  std::vector<size_t> permutation(outofbag);
  
  arma::vec permutation_importance(x.n_cols, arma::fill::zeros);
  for(size_t var = 0; var < x.n_cols; ++var) {
    std::shuffle(permutation.begin(), permutation.end(), rand);
    int errors = 0;
    for(size_t i = 0; i < outofbag.size(); ++i) {
      size_t pred = predictPermuted(outofbag[i], permutation[i], var);
      if(pred != y[outofbag[i]]) errors++;
    }
    
    permutation_importance[var] = (double)errors / y.n_elem - base_error;
  }
  
  return permutation_importance;
}

arma::uword Tree::predictPermuted(size_t smp, size_t smp_permuted, size_t var) {
  size_t node = 0;
  while(true) {
    if(split_child_left[node] == 0) {
      return split_var[node];
    } else {
      double value;
      if(split_var[node] == var) {
        value = x.at(smp_permuted, split_var[node]);
      } else {
        value = x.at(smp, split_var[node]);
      }
      
      if(value < split_value[node]) {
        node = split_child_left[node];
      } else {
        node = split_child_right[node];
      }
    }
  }
}
