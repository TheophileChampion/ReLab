#include "deque.hpp"
#include "serialize.hpp"
#include "debug.hpp"

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

template<class T>
void Deque<T>::load(std::istream &checkpoint) {

    // Load the deque from the checkpoint.
    this->max_size = serialize::load_value<int>(checkpoint);
    int size = serialize::load_value<int>(checkpoint);
    for (auto i = 0; i < size; i++) {
        this->push_back(serialize::load_value<T>(checkpoint));
    }
}

template<class T>
void Deque<T>::save(std::ostream &checkpoint) {

    // Save the deque in the checkpoint.
    serialize::save_value(this->max_size, checkpoint);
    serialize::save_value(static_cast<int>(this->size()), checkpoint);
    for (T element : *this) {
        serialize::save_value(element, checkpoint);
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

template<>
void Deque<bool>::print() {
    std::cout << "Deque(max_size: " << this->max_size << ", values: [";
    int i = 0;
    for (bool element : *this) {
        if (i != 0) {
            std::cout << " ";
        }
        debug::print_bool(element);
        ++i;
    }
    std::cout << "])" << std::endl;
}
