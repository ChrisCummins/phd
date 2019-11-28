/*
 * Implement a MyQueue class which implements a queue using two
 * stacks.
 */
#include "./ctci.h"

#include <stack>

//
// Pass objects between two stacks such that the right hand stack is
// full of objects waiting to be popped, and the left hand stack is
// objects waiting to be moved to the right hand side.
//
// push(): O(1) time, O(1) space
// pop(): O(n) time, O(1) space
//
template <typename T>
class MyQueue {
 public:
  MyQueue() {}

  // Append to queue.
  void push(const T& val) { left.push(val); }

  void push(T&& val) { left.push(std::move(val)); }

  // Remove from front of queue.
  T pop() {
    if (right.empty()) {
      while (!left.empty()) {
        right.push(left.top());
        left.pop();
      }
    }

    auto tmp = right.top();
    right.pop();
    return tmp;
  }

  bool empty() const { return right.empty() && left.empty(); }

 private:
  std::stack<T> left;
  std::stack<T> right;
};

///////////
// Tests //
///////////

TEST(StackQueue, MyQueue) {
  MyQueue<int> q1;

  q1.push(1);
  q1.push(2);
  q1.push(3);

  ASSERT_EQ(1, q1.pop());
  ASSERT_EQ(2, q1.pop());

  q1.push(4);
  q1.push(5);
  q1.push(6);

  ASSERT_EQ(3, q1.pop());
  ASSERT_EQ(4, q1.pop());
  ASSERT_EQ(5, q1.pop());
  ASSERT_EQ(6, q1.pop());

  ASSERT_TRUE(q1.empty());
}

CTCI_MAIN();
