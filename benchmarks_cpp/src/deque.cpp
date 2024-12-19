#include "deque.hpp"

template<class T>
Deque<T>::Deque(int max_size) : max_size(max_size) {}

template<class T>
void Deque<T>::push_back(T element) {
    if (this->max_size >= 0 and static_cast<int>(this->size()) >= this->max_size) {
        this->pop_front();
    }
    this->std::deque<T>::push_back(std::move(element));
}

template<class T>
void Deque<T>::push_front(T element) {
    if (this->max_size >= 0 and static_cast<int>(this->size()) >= this->max_size) {
        this->pop_back();
    }
    this->std::deque<T>::push_front(std::move(element));
}

template<class T>
T Deque<T>::get(int index) {
    return (*this)[index];
}

template<class T>
void Deque<T>::clear() {
    this->std::deque<T>::clear();
}

template<class T>
void Deque<T>::pop_back() {
    this->std::deque<T>::pop_back();
}

template<class T>
void Deque<T>::pop_front() {
    this->std::deque<T>::pop_front();
}

template<class T>
std::uint64_t Deque<T>::size() {
    return this->std::deque<T>::size();
}

