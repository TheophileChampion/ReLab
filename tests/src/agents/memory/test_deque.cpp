#include <gtest/gtest.h>
#include <torch/extension.h>
#include "agents/memory/test_deque.hpp"
#include "helpers/deque.hpp"
#include "relab_test.hpp"

namespace relab::test::agents::memory {

    DequeParameters::DequeParameters(const std::initializer_list<int> &elements, int max_size, int n_pops, int length)
        : elements(elements), max_size(max_size), n_pops(n_pops), length(length) {}

    DequeParameters::DequeParameters() : DequeParameters({}, 0, 0, 0) {}

    void TestDeque::compare(Deque<int> &queue, const std::deque<int> &result) {
        EXPECT_EQ(queue.size(), result.size());
        for (size_t i = 0; i < queue.size(); i++) {
            EXPECT_EQ(queue[i], result[i]);
        }
    }

    std::deque<int> TestDeque::getResult(const DequeParameters &params, const PushType &push_type, const PopType &pop_type) {

        // Create the result deque.
        std::deque<int> result;

        // Add all elements to the result deque.
        for (auto element : params.elements) {
            if (push_type == PushType::FRONT) {
                result.push_front(element);
            } else {
                result.push_back(element);
            }
        }

        // Remove elements from the result deque to ensure the queue size is inferior to the max size.
        while (static_cast<int>(result.size()) > params.max_size) {
            if (push_type == PushType::FRONT) {
                result.pop_back();
            } else {
                result.pop_front();
            }
        }

        // Remove elements from the result deque.
        if (pop_type == PopType::NO_POP) {
            return result;
        }
        for (int i = 0; i < params.n_pops; i++) {
            if (pop_type == PopType::FRONT) {
                result.pop_front();
            } else {
                result.pop_back();
            }
        }
        return result;
    }

    void TestDeque::SetUp() {

        // Create a deque.
        this->params = GetParam();
        this->queue = std::make_unique<Deque<int>>(params.max_size);
    }

    TEST_P(TestDeque, TestClear) {

        // Add all elements to the end of the double ended queue.
        for (auto element : params.elements) {
            queue->push_back(element);
        }

        // Clear the double ended queue.
        queue->clear();

        // Check that the queue matches the expected result.
        std::deque<int> result;
        TestDeque::compare(*queue, result);
    }

    TEST_P(TestDeque, TestPushBack) {

        // Add all elements to the end of the double ended queue.
        for (auto element : params.elements) {
            queue->push_back(element);
        }

        // Check that the queue matches the expected result.
        auto result = TestDeque::getResult(params, PushType::BACK, PopType::NO_POP);
        TestDeque::compare(*queue, result);
    }

    TEST_P(TestDeque, TestPushFront) {

        // Add all elements to the front of the double ended queue.
        for (auto element : params.elements) {
            queue->push_front(element);
        }

        // Check that the queue matches the expected result.
        auto result = TestDeque::getResult(params, PushType::FRONT, PopType::NO_POP);
        TestDeque::compare(*queue, result);
    }

    TEST_P(TestDeque, TestPopFront) {

        // Add all elements to the end of the double ended queue.
        for (auto element : params.elements) {
            queue->push_front(element);
        }

        // Remove elements from the front of the double ended queue.
        for (int i = 0; i < params.n_pops; i++) {
            queue->pop_front();
        }

        // Check that the queue matches the expected result.
        auto result = TestDeque::getResult(params, PushType::FRONT, PopType::FRONT);
        TestDeque::compare(*queue, result);
    }

    TEST_P(TestDeque, TestPopBack) {

        // Add all elements to the end of the double ended queue.
        for (auto element : params.elements) {
            queue->push_front(element);
        }

        // Remove elements from the front of the double ended queue.
        for (int i = 0; i < params.n_pops; i++) {
            queue->pop_back();
        }

        // Check that the queue matches the expected result.
        auto result = TestDeque::getResult(params, PushType::FRONT, PopType::BACK);
        TestDeque::compare(*queue, result);
    }

    TEST_P(TestDeque, TestSize) {

        // Add all elements to the end of the double ended queue.
        for (auto element : params.elements) {
            queue->push_back(element);
        }

        // Check that the queue's size is correct.
        EXPECT_EQ(queue->size(), params.length);
    }

    INSTANTIATE_TEST_SUITE_P(UnitTests, TestDeque, testing::Values(
        DequeParameters({5, 1, 1}, 2, 1, 2), DequeParameters({}, 4, 0, 0),
        DequeParameters({1, 2, 3, 4}, 4, 3, 4), DequeParameters({10, 2}, 4, 2, 2)
    ));
}
