// Copyright 2025 Theophile Champion. No Rights Reserved.

#include <utility>
#include <iostream>
#include "helpers/deque.hpp"
#include "helpers/serialize.hpp"
#include "helpers/debug.hpp"

namespace relab::helpers {

template<class T>
Deque<T>::Deque(int max_size) : max_size(max_size) {}

template<class T>
void Deque<T>::push_back(T element) {
    if (this->max_size >= 0 && static_cast<int>(this->size()) >= this->max_size) {
        this->pop_front();
    }
    this->std::deque<T>::push_back(std::move(element));
}

template<class T>
void Deque<T>::push_front(T element) {
    if (this->max_size >= 0 && static_cast<int>(this->size()) >= this->max_size) {
        this->pop_back();
    }
    this->std::deque<T>::push_front(std::move(element));
}

template<class T>
T Deque<T>::get(int index) {
    return (*this)[index];
}

template<class T>
void Deque<T>::load(std::istream &checkpoint) {

    // Load the deque from the checkpoint.
    this->max_size = load_value<int>(checkpoint);
    int size = load_value<int>(checkpoint);
    for (auto i = 0; i < size; i++) {
        this->push_back(load_value<T>(checkpoint));
    }
}

template<class T>
void Deque<T>::save(std::ostream &checkpoint) {

    // Save the deque in the checkpoint.
    save_value(this->max_size, checkpoint);
    save_value(static_cast<int>(this->size()), checkpoint);
    for (T element : *this) {
        save_value(element, checkpoint);
    }
}

template<class T>
void Deque<T>::print() {
    std::cout << "Deque(max_size: " << this->max_size << ", values: [";
    int i = 0;
    for (T element : *this) {
        if (i != 0) {
            std::cout << " ";
        }
        std::cout << element;
        ++i;
    }
    std::cout << "])" << std::endl;
}

// @cond IGNORED_BY_DOXYGEN
template<>
void Deque<bool>::print() {
    std::cout << "Deque(max_size: " << this->max_size << ", values: [";
    int i = 0;
    for (bool element : *this) {
        if (i != 0) {
            std::cout << " ";
        }
        print_bool(element);
        ++i;
    }
    std::cout << "])" << std::endl;
}
// @endcond

template<class Type>
bool operator==(const Deque<Type> &lhs, const Deque<Type> &rhs) {
    if (lhs.max_size != rhs.max_size || lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); i++) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

// Explicit instantiation of double ended queue.
template bool operator==(const Deque<int> &lhs, const Deque<int> &rhs);
template bool operator==(const Deque<float> &lhs, const Deque<float> &rhs);
template bool operator==(const Deque<bool> &lhs, const Deque<bool> &rhs);
}  // namespace relab::helpers
