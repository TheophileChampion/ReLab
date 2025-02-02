// Copyright 2025 Theophile Champion. No Rights Reserved.

#include "agents/memory/priority_tree.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include "helpers/debug.hpp"
#include "helpers/serialize.hpp"
#include "helpers/torch.hpp"

using namespace torch::indexing;
using namespace relab::helpers;

namespace relab::agents::memory::impl {

PriorityTree::PriorityTree(int capacity, float initial_priority,
                           int n_children) {
  // Store the priority tree parameters.
  this->initial_priority = initial_priority;
  this->capacity = capacity;
  this->n_children = n_children;

  // Robust computation of the trees' depth.
  this->depth = std::floor(std::log(this->capacity) / std::log(n_children));
  if (static_cast<int>(std::pow(n_children, this->depth)) < this->capacity) {
    this->depth += 1;
  }

  // Create a tensor of priorities, an empty sum-tree and an empty max-tree.
  this->priorities = torch::zeros({this->capacity});
  this->sum_tree = this->createSumTree(this->depth, n_children);
  this->max_tree = this->createMaxTree(this->depth, n_children);
  this->current_id = 0;
  this->need_refresh_all = true;
}

SumTree PriorityTree::createSumTree(int depth, int n_children) {
  SumTree tree;

  for (auto i = depth - 1; i >= 0; i--) {
    int n = std::pow(n_children, i);
    std::vector<double> row(n);
    tree.push_back(std::move(row));
  }
  return tree;
}

MaxTree PriorityTree::createMaxTree(int depth, int n_children) {
  MaxTree tree;

  for (auto i = depth - 1; i >= 0; i--) {
    tree.push_back(torch::zeros({static_cast<int>(std::pow(n_children, i))}));
  }
  return tree;
}

double PriorityTree::sum() {
  if (this->current_id == 0) {
    return 0;
  }
  return this->sum_tree[this->sum_tree.size() - 1][0];
}

float PriorityTree::max() {
  if (this->current_id == 0) {
    return this->initial_priority;
  }
  return this->max_tree[this->max_tree.size() - 1][0].item<float>();
}

void PriorityTree::clear() {
  this->current_id = 0;
  this->need_refresh_all = true;
  this->priorities = torch::zeros({this->capacity});
  this->sum_tree = this->createSumTree(this->depth, this->n_children);
  this->max_tree = this->createMaxTree(this->depth, this->n_children);
}

int PriorityTree::size() { return std::min(this->current_id, this->capacity); }

void PriorityTree::append(float priority) {
  int idx = this->current_id % this->capacity;
  float old_priority = this->priorities[idx].item<float>();

  // Add a new priority to the list of priorities.
  this->priorities[idx] = priority;
  this->current_id += 1;
  this->updateMaxTree(idx, old_priority);
  this->updateSumTree(idx, old_priority);

  // Check if the full sum tree must be refreshed.
  if (this->max() != this->initial_priority and
      this->need_refresh_all == true) {
    this->refreshAllSumTree();
    this->need_refresh_all = false;
  }
}

float PriorityTree::get(int index) {
  return this->priorities[this->internalIndex(index)].item<float>();
}

void PriorityTree::set(int index, float priority) {
  int idx = this->internalIndex(index);
  float old_priority = this->priorities[idx].item<float>();

  // Replace the old priority with the new priority.
  this->priorities[idx] = priority;
  this->updateMaxTree(idx, old_priority);
  this->updateSumTree(idx, old_priority);

  // Check if the full sum tree must be refreshed.
  if (this->max() != this->initial_priority and
      this->need_refresh_all == true) {
    this->refreshAllSumTree();
    this->need_refresh_all = false;
  }
}

int PriorityTree::internalIndex(int index) {
  if (this->current_id >= this->capacity) {
    index += this->current_id;
  }
  index %= this->capacity;
  return (index >= 0) ? index : index + this->size();
}

int PriorityTree::externalIndex(int index) {
  if (this->current_id >= this->capacity) {
    index -= (this->current_id % this->capacity);
  }
  index %= this->capacity;
  return (index >= 0) ? index : index + this->size();
}

torch::Tensor PriorityTree::sampleIndices(int n) {
  // Sample priorities between zero and the sum of priorities.
  torch::Tensor sampled_priorities =
      torch::rand({n}) * static_cast<float>(this->sum());

  // Sample 'n' indices with a probability proportional to their priorities.
  torch::Tensor indices = torch::zeros({n}, torch::kInt64);
  for (auto i = 0; i < n; i++) {
    float priority = sampled_priorities.index({i}).item<float>();
    indices.index_put_({i}, static_cast<long>(this->towerSampling(priority)));
  }
  return indices;
}

int PriorityTree::towerSampling(float priority) {
  // If the priority is larger than the sum of priorities, return the index of
  // the last element.
  if (priority > this->sum()) {
    return this->externalIndex(this->size() - 1);
  }

  // Go down the sum-tree until the leaf node is reached.
  float new_priority = 0;
  int index = 0;
  for (int level = this->depth - 2; level >= -1; level--) {

    // Iterate over the children of the current node, keeping track of the sum
    // of priorities.
    float total = 0;
    for (auto i = 0; i < this->n_children; i++) {

      // Get the priority of the next child.
      int child_index = this->n_children * index + i;
      if (level == -1) {
        new_priority = this->priorities[child_index].item<float>();
      } else {
        new_priority = this->sum_tree[level][child_index];
      }

      // If the priority is about to be superior to the total, stop iterating
      // over the children.
      if (priority <= total + new_priority) {
        index = child_index;
        priority -= total;
        break;
      }

      // Otherwise, increase the sum of priorities.
      total += new_priority;
    }
  }

  // Return the element index corresponding to the sampled priority.
  return this->externalIndex(index);
}

int PriorityTree::parentIndex(int idx) {
  return (idx < 0) ? idx : idx / this->n_children;
}

void PriorityTree::updateSumTree(int index, float old_priority) {
  // Compute the parent index.
  int parent_index = this->parentIndex(index);

  // Go up the tree until the root node is reached.
  int depth = 0;
  float new_priority = this->priorities[index].item<float>();
  while (depth < this->depth) {

    // Update the sums in the sum-tree.
    this->sum_tree[depth][parent_index] += new_priority - old_priority;

    // Update parent indices and tree depth.
    depth += 1;
    parent_index = this->parentIndex(parent_index);
  }
}

void PriorityTree::refreshAllSumTree() {
  // Fill the sum-tree with zeros.
  this->sum_tree = this->createSumTree(this->depth, this->n_children);

  // Iterate over all the priorities.
  for (auto index = 0; index < this->size(); index++) {

    // Compute the parent index and current priority.
    int parent_index = this->parentIndex(index);
    float priority = this->priorities[index].item<float>();

    // Go up the tree until the root node is reached.
    int depth = 0;
    while (depth < this->depth) {

      // Update the sums in the sum-tree.
      this->sum_tree[depth][parent_index] += priority;

      // Update parent indices and tree depth.
      depth += 1;
      parent_index = this->parentIndex(parent_index);
    }
  }
}

void PriorityTree::updateMaxTree(int index, float old_priority) {
  // Compute the parent index and the old priority.
  int parent_index = this->parentIndex(index);
  float new_priority = this->priorities[index].item<float>();

  // Go up the tree until the root node is reached.
  int depth = 0;
  while (depth < this->depth) {

    // Update the maximum values in the max-tree.
    float parent_value = this->max_tree[depth][parent_index].item<float>();
    if (parent_value == old_priority) {
      this->max_tree[depth][parent_index] = this->maxChildValue(
          depth, parent_index, index, old_priority, new_priority);
    } else if (parent_value < new_priority) {
      this->max_tree[depth][parent_index] = new_priority;
    } else {
      break;
    }

    // Update parent indices and tree depth.
    depth += 1;
    parent_index = this->parentIndex(parent_index);
  }
}

float PriorityTree::maxChildValue(int depth, int parent_index, int index,
                                  float old_priority, float new_priority) {
  int first_child = this->n_children * parent_index;
  auto slice = Slice(first_child, first_child + this->n_children);
  float max_value = 0;

  if (depth == 0) {
    torch::Tensor children = this->priorities.index({slice});
    max_value = children.max().item<float>();
  } else {
    max_value = this->max_tree[depth - 1].index({slice}).max().item<float>();
  }
  return max_value;
}

std::string PriorityTree::maxTreeToStr(int max_n_elements) {
  float (*get)(MaxTree, int, int) = [](MaxTree tree, int i, int j) {
    return tree[i].index({j}).item<float>();
  };
  return this->treeToStr(this->max_tree, get, max_n_elements);
}

std::string PriorityTree::sumTreeToStr(int max_n_elements) {
  double (*get)(SumTree, int, int) = [](SumTree tree, int i, int j) {
    return tree[i][j];
  };
  return this->treeToStr(this->sum_tree, get, max_n_elements);
}

template <class Tree, class T>
std::string PriorityTree::treeToStr(Tree tree, T (*get)(Tree, int, int),
                                    int max_n_elements, int precision) {
  int n = static_cast<int>(tree.size());
  if (max_n_elements == -1) {
    max_n_elements = n;
  }
  std::ostringstream out;
  out.precision(precision);
  out << std::fixed << "[";

  // Iterate over all sub-lists.
  for (auto i = 0; i < n; i++) {

    // Open the bracket in the string.
    out << ((i != 0) ? ", [" : "[");

    // Iterate over all elements.
    int m = std::pow(this->n_children, this->depth - 1 - i);
    int max_j = (max_n_elements == -1) ? m : std::min(m, max_n_elements);
    for (auto j = 0; j < max_j; j++) {

      // Add all elements to the string.
      if (j != 0)
        out << ", ";
      out << get(tree, i, j);
    }

    // Close the bracket in the string, adding an ellipse if only part of inner
    // vector was displayed.
    if (max_j != m) {
      out << ((max_n_elements != 0) ? " ..." : "...");
    }
    out << "]";
  }
  out << "]";
  return out.str();
}

void PriorityTree::load(std::istream &checkpoint) {
  // Load the priority tree from the checkpoint.
  this->initial_priority = load_value<float>(checkpoint);
  this->capacity = load_value<int>(checkpoint);
  this->n_children = load_value<int>(checkpoint);
  this->depth = load_value<int>(checkpoint);
  this->current_id = load_value<int>(checkpoint);
  this->need_refresh_all = load_value<bool>(checkpoint);
  this->priorities = load_tensor<float>(checkpoint);
  this->sum_tree.clear();
  this->sum_tree.reserve(this->depth);
  for (auto i = 0; i < this->depth; i++) {
    this->sum_tree.push_back(load_vector<double>(checkpoint));
  }
  this->max_tree = load_vector<torch::Tensor, float>(checkpoint);
}

void PriorityTree::save(std::ostream &checkpoint) {
  // Save the priority tree in the checkpoint.
  save_value(this->initial_priority, checkpoint);
  save_value(this->capacity, checkpoint);
  save_value(this->n_children, checkpoint);
  save_value(this->depth, checkpoint);
  save_value(this->current_id, checkpoint);
  save_value(this->need_refresh_all, checkpoint);
  save_tensor<float>(this->priorities, checkpoint);
  for (auto i = 0; i < this->depth; i++) {
    save_vector(this->sum_tree[i], checkpoint);
  }
  save_vector<torch::Tensor, float>(this->max_tree, checkpoint);
}

void PriorityTree::print(bool verbose, const std::string &prefix) {
  // Display the most important information about the data buffer.
  std::cout << "PriorityTree[initial_priority: " << this->initial_priority
            << ", capacity: " << this->capacity
            << ", n_children: " << this->n_children
            << ", depth: " << this->depth
            << ", current_id: " << this->current_id << ", need_refresh_all: ";
  print_bool(this->need_refresh_all);
  std::cout << "]" << std::endl;

  // Display optional information about the data buffer.
  if (verbose == true) {
    std::cout << prefix << " #-> priorities = ";
    print_tensor<float>(this->priorities, 10);
    std::cout << prefix << " #-> sum_tree = " << this->sumTreeToStr(3)
              << std::endl;
    std::cout << prefix << " #-> max_tree = " << this->maxTreeToStr(3)
              << std::endl;
  }
}

bool operator==(const PriorityTree &lhs, const PriorityTree &rhs) {
  // Check that all attributes of standard types and container sizes are
  // identical.
  if (lhs.initial_priority != rhs.initial_priority ||
      lhs.capacity != rhs.capacity || lhs.n_children != rhs.n_children ||
      lhs.depth != rhs.depth || lhs.current_id != rhs.current_id ||
      lhs.need_refresh_all != rhs.need_refresh_all ||
      lhs.sum_tree.size() != rhs.sum_tree.size() ||
      lhs.max_tree.size() != rhs.max_tree.size()) {
    return false;
  }

  // Compare the priorities.
  if (!tensorsAreEqual(lhs.priorities, rhs.priorities)) {
    return false;
  }

  // Compare the sum-trees.
  for (size_t i = 0; i < lhs.sum_tree.size(); i++) {
    if (lhs.sum_tree[i].size() != rhs.sum_tree[i].size()) {
      return false;
    }
    for (size_t j = 0; j < lhs.sum_tree[i].size(); j++) {
      if (lhs.sum_tree[i][j] != rhs.sum_tree[i][j]) {
        return false;
      }
    }
  }

  // Compare the max-trees.
  for (size_t i = 0; i < lhs.max_tree.size(); i++) {
    if (!tensorsAreEqual(lhs.max_tree[i], rhs.max_tree[i])) {
      return false;
    }
  }
  return true;
}
}  // namespace relab::agents::memory::impl
