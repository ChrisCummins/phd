/*
 * Image a (literal) stack of plates. If the stack gets to high, it
 * might topple. Therefore, in real life, we would like start a new
 * stack when the previous stack exceeds some threshold. Implement a
 * data structure SetOfStacks that mimics this. SetOfStacks should be
 * composed of several stacks and should create a new stack once the
 * previous one exeeds capacity. SetOfStacks.push() and
 * SetOfStacks.pop() should behave identically to a single stack (that
 * is, pop() should return the same values as it would if there were
 * just a single stack).
 *
 * Follow up:
 *
 * Implement a function popAt(int index) which performs a pop
 * operation on a specific sub-stack.
 */
#include "./ctci.h"

#include <iostream>
#include <list>
#include <numeric>
#include <stack>
#include <stdexcept>
#include <vector>

template <typename T, size_t stack_size = 2>
class SetOfStacks {
 public:
  using stack_type = std::stack<T>;
  using stacks_list = std::vector<stack_type>;

  SetOfStacks() : _stacks({stack_type{}}) {}

  SetOfStacks(std::initializer_list<T> il) : SetOfStacks() {
    auto it = il.begin();
    while (it != il.end()) push(*it++);
  }

  void push(const T& val) {
    auto& stack = _stacks.back();
    stack.push(val);

    if (stack.size() == stack_size) _stacks.emplace_back();
  }

  void push(T&& val) {
    auto& stack = _stacks.back();
    stack.push(std::move(val));

    if (stack.size() == stack_size) _stacks.emplace_back();
  }

  T pop() {
    while (_stacks.back().empty() && !_stacks.empty())
      _stacks.erase(_stacks.end() - 1);

    if (_stacks.empty()) _stacks.push_back(stack_type{});

    return popAt(static_cast<int>(_stacks.size() - 1u));
  }

  T popAt(int index) {
    size_t i = static_cast<size_t>(index);

    if (index < 0 || i >= _stacks.size() || _stacks[i].empty())
      throw std::out_of_range("SetOfStacks::popAt()");

    auto& stack = _stacks[i];
    auto& val = stack.top();
    stack.pop();

    return val;
  }

  bool empty() const {
    return std::all_of(_stacks.begin(), _stacks.end(),
                       [](auto& s) { return s.empty(); });
  }

  size_t size() const {
    return std::accumulate(
        _stacks.begin(), _stacks.end(), size_t{0},
        [](auto acc, const auto& stack) { return acc + stack.size(); });
  }

  friend std::ostream& operator<<(std::ostream& out, const SetOfStacks& s) {
    for (auto& stack : s._stacks) {
      out << "stack(" << stack.size() << ") ";
      if (!stack.empty()) out << "top = " << stack.top();
      out << std::endl;
    }
    return out;
  }

 private:
  stacks_list _stacks;
};

///////////
// Tests //
///////////

TEST(SetOfStacks, pop) {
  SetOfStacks<int> s{5, 4, 3, 2, 1};

  ASSERT_EQ(5u, s.size());
  ASSERT_FALSE(s.empty());

  s.push(6);
  ASSERT_EQ(6u, s.size());

  s.push(7);
  ASSERT_EQ(7u, s.size());

  ASSERT_EQ(7, s.pop());
  ASSERT_EQ(6, s.pop());

  ASSERT_EQ(2, s.popAt(1));

  ASSERT_EQ(1, s.pop());
  ASSERT_FALSE(s.empty());

  ASSERT_EQ(3, s.pop());
  ASSERT_FALSE(s.empty());

  ASSERT_EQ(4, s.pop());
  ASSERT_FALSE(s.empty());

  ASSERT_EQ(5, s.pop());
  ASSERT_TRUE(s.empty());
}

CTCI_MAIN();
