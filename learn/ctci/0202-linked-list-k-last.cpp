/*
 * Implement an algorithm to find the kth to last element of a single liked list.
 */
#include "./ctci.h"

#include <vector>
#include <forward_list>

template<typename T>
typename std::forward_list<T>::const_iterator
k_last_elem(const std::forward_list<T> lst,
            const size_t k) {
    auto first = lst.begin(), last = lst.end();
    std::vector<typename std::forward_list<T>::const_iterator> tmp;

    while (first != last) {
        tmp.push_back(first++);
        if (tmp.size() > k + 1)
            tmp.erase(tmp.begin());
    }

    return *tmp.begin();
}

TEST(challenge, basic) {
    std::forward_list<int> l{1, 2, 3, 4, 5};

    ASSERT_EQ(4, *k_last_elem(l, 1));
    ASSERT_EQ(3, *k_last_elem(l, 2));
    ASSERT_EQ(2, *k_last_elem(l, 3));
    ASSERT_EQ(1, *k_last_elem(l, 4));
}

CTCI_MAIN();
