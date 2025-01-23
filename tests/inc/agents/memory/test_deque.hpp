#ifndef TEST_DEQUE_HPP
#define TEST_DEQUE_HPP

#include <gtest/gtest.h>
#include <memory>
#include "helpers/deque.hpp"
#include "agents/memory/experience.hpp"

namespace relab::test::agents::memory::impl {

    using namespace relab::helpers;

    /**
     * A class storing the parameters of the deque tests.
     */
    class DequeParameters {

    public:

        std::vector<int> elements;
        int max_size;
        int n_pops;
        int length;

    public:

        /**
         * Create a structure storing the parameters of the deque tests.
         * @param elements the elements to push in the deque
         * @param max_size the maximum number of elements the deque can contain
         * @param n_pops the number of elements to pop from the queue
         * @param length the expected length of the deque at the end of the test
         */
        DequeParameters(const std::initializer_list<int> &elements, int max_size, int n_pops, int length);

        /**
         * Create a structure storing the parameters of the deque tests.
         */
        DequeParameters();
    };

    /**
     * Enumeration of type of push, used by TestDeque::getResult.
     */
    enum class PushType {
        FRONT = 0,  // Push at the back of the queue.
        BACK = 1  // Push at the front of the queue.
    };

    /**
     * Enumeration of type of pop, used by TestDeque::getResult.
     */
    enum class PopType {
        NO_POP = 0,  // Don't pop any element from the queue.
        FRONT = 1,  // Pop element at the back of the queue.
        BACK = 2  // Pop element at the front of the queue.
    };

    /**
     * A fixture class for testing the deque.
     */
    class TestDeque : public testing::TestWithParam<DequeParameters> {

    public:

        DequeParameters params;
        std::unique_ptr<Deque<int>> queue;

    public:

        /**
         * Compare the inpt queue to the expected result.
         * @param queue the input queue
         * @param result the expected result
         */
        static void compare(Deque<int> &queue, const std::deque<int> &result);

        /**
         * Create the expected result of the test.
         * @param params the test parameters
         * @param push_type the type of push performed in the test
         * @param pop_type the type of pop performed in the test
         * @return the expected result of the test
         */
        static std::deque<int> getResult(const DequeParameters &params, const PushType &push_type, const PopType &pop_type);

        /**
         * Setup of th fixture class before calling a unit test.
         */
        void SetUp();
    };
}

namespace relab::test::agents::memory {
    using impl::DequeParameters;
    using impl::PushType;
    using impl::PopType;
    using impl::TestDeque;
}

#endif //TEST_DEQUE_HPP
