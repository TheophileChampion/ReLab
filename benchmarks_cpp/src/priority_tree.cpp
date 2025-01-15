#include "priority_tree.hpp"
#include <cmath>

using namespace torch::indexing;

PriorityTree::PriorityTree(int capacity, float initial_priority, int n_children) {

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
        std::vector<long double> row(n);
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

long double PriorityTree::sum() {
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

int PriorityTree::size() {
    return std::min(this->current_id, this->capacity);
}

void PriorityTree::append(float priority) {

    int idx = this->current_id % this->capacity;
    float old_priority = this->priorities[idx].item<float>();

    // Add a new priority to the list of priorities.
    this->priorities[idx] = priority;
    this->current_id += 1;
    this->updateMaxTree(idx, old_priority);
    this->updateSumTree(idx, old_priority);

    // Check if the full sum tree must be refreshed.
    if (this->max() != this->initial_priority and this->need_refresh_all == true) {
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
    if (this->max() != this->initial_priority and this->need_refresh_all == true) {
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
    torch::Tensor sampled_priorities = torch::rand({n}) * static_cast<float>(this->sum());

    // Sample 'n' indices with a probability proportional to their priorities.
    torch::Tensor indices = torch::zeros({n}, torch::kInt64);
    for (auto i = 0; i < n; i++) {
        float priority = sampled_priorities.index({i}).item<float>();
        indices.index_put_({i}, static_cast<long>(this->towerSampling(priority)));
    }
    return indices;
}

int PriorityTree::towerSampling(float priority) {

    // If the priority is larger than the sum of priorities, return the index of the last element.
    if (priority > this->sum()) {
        return this->externalIndex(this->size() - 1);
    }

    // Go down the sum-tree until the leaf node is reached.
    float new_priority = 0;
    int index = 0;
    for (int level = this->depth - 2; level >= -1; level--) {

        // Iterate over the children of the current node, keeping track of the sum of priorities.
        float total = 0;
        for (auto i = 0; i < this->n_children; i++) {

            // Get the priority of the next child.
            int child_index = this->n_children * index + i;
            if (level == -1) {
                new_priority = this->priorities[child_index].item<float>();
            } else {
                new_priority = static_cast<float>(this->sum_tree[level][child_index]);
            }

            // If the priority is about to be superior to the total, stop iterating over the children.
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
            this->max_tree[depth][parent_index] = this->maxChildValue(depth, parent_index, index, old_priority, new_priority);
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

float PriorityTree::maxChildValue(int depth, int parent_index, int index, float old_priority, float new_priority) {
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

std::string PriorityTree::maxTreeToStr() {

    int n = static_cast<int>(this->max_tree.size());
    std::string tree_str = "[";

    // Iterate over all sub-lists.
    for (auto i = 0; i < n; i++) {

        // Open the bracket in the string.
        if (i != 0)
            tree_str += ", [";
        else
            tree_str += "[";

        // Iterate over all elements.
        int m = this->max_tree[i].numel();
        for (auto j = 0; j < m; j++) {

            // Add all elements to the string.
            if (j != 0)
                tree_str += ", ";
            tree_str += this->toString(this->max_tree[i].index({j}).item<float>());
        }

        // Close the bracket in the string.
        tree_str += "]";
    }
    return tree_str + "]";
}

std::string PriorityTree::sumTreeToStr() {

    int n = static_cast<int>(this->sum_tree.size());
    std::string tree_str = "[";

    // Iterate over all sub-lists.
    for (auto i = 0; i < n; i++) {

        // Open the bracket in the string.
        if (i != 0)
            tree_str += ", [";
        else
            tree_str += "[";

        // Iterate over all elements.
        int m = static_cast<int>(this->sum_tree[i].size());
        for (auto j = 0; j < m; j++) {

            // Add all elements to the string.
            if (j != 0)
                tree_str += ", ";
            tree_str += this->toString(static_cast<float>(this->sum_tree[i][j]));
        }

        // Close the bracket in the string.
        tree_str += "]";
    }
    return tree_str + "]";
}

std::string PriorityTree::toString(float value, int precision) {
    std::ostringstream out;
    out.precision(precision);
    out << std::fixed << value;
    return out.str();
}
