#include "Tree.h"
#include <cmath>
#include <algorithm>
#include <array>
#include <iterator>

Tree::Tree(
  const arma::mat &x, const arma::uvec &y,
  unsigned int mtry,
  bool replace, double sample_fraction,
  unsigned int seed
) : x(x), y(y), mtry(mtry), replace(replace), sample_fraction(sample_fraction), rand(seed)
{
  y_levels = arma::unique(y);
}

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
  
  importance = arma::vec(x.n_cols, arma::fill::zeros);
    
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
      if(x(smp, split_var[node]) < split_value[node]) {
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
  for(size_t var : candidate_vars) {
    arma::vec values = arma::unique(x.col(var));
    for(double value : values) {
      std::array<std::vector<size_t>,2> assign;
      for(size_t smp : samples[split_index]) {
        if(x(smp, var) < value) {
          assign[0].push_back(smp);
        } else {
          assign[1].push_back(smp);
        }
      }
      
      if(assign[0].size() == 0 || assign[1].size() == 0) continue;
      
      double decrease = 0;
      for(const std::vector<size_t> &subtree : assign) {
        double sum = 0;
        std::vector<int> counts(y_levels.n_elem, 0);
        for(size_t smp : subtree) {
          arma::uword value = y[smp];
          counts[value]++;
        }
        for(int count : counts) {
          sum += count * count;
        }
        decrease += sum / (double)subtree.size();
      }
      
      if(decrease > best_decrease) {
        best_decrease = decrease;
        best_var = var;
        best_value = value;
      }
    }
  }
  
  if(best_decrease <= 0) {
    arma::uvec y_indices(samples[split_index].size());
    std::copy(samples[split_index].begin(), samples[split_index].end(), y_indices.begin());
    
    arma::uword y_max = arma::max(y(y_indices));
    arma::uvec sp = arma::linspace<arma::uvec>(0, y_max, y_max+1);
    arma::uvec y_hist = arma::hist(y(y_indices), sp);
    arma::uword estimate = y_hist.index_max();
    
    split_var[split_index] = estimate;
    return;
  }
  
  split_var[split_index] = best_var;
  split_value[split_index]  = best_value;
  
  size_t left_index = createNode();
  size_t right_index = createNode();
  split_child_left[split_index] = left_index;
  split_child_right[split_index] = right_index;
  
  for(size_t smp : samples[split_index]) {
    if(x(smp, best_var) < best_value) {
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
  std::vector<int> counts(y_levels.n_elem, 0);
  for(size_t smp : samples[split_index]) {
    arma::uword value = y[smp];
    counts[value]++;
  }
  double sum = 0;
  for(int count : counts) {
    sum += count * count;
  }
  
  double node_decrease = decrease - sum / (double)samples[split_index].size();
  importance[var] += node_decrease;
}
