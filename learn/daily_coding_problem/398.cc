// This problem was asked by Amazon.
//
// Given a linked list and an integer k, remove the k-th node from the end of
// the list and return the head of the list.
//
// k is guaranteed to be smaller than the length of the list.
//
// Do this in one pass.

#include "labm8/cpp/logging.h"
#include "labm8/cpp/test.h"

// My list definition. Singly-linked.
struct L {
  L* next;

  L() : next(nullptr) {}
};

// "Removes" k-th to last node from the list. Doesn't free the memory.
L* Remove(L* list, int k) {
  // Check arguments.
  if (list == nullptr || k < 0) {
    return nullptr;
  }

  L* head = list;
  L* p = head;
  L* j = head;
  L* kp = head;

  int i = 0;
  while (p) {
    ++i;
    LOG(INFO) << "i = " << i;
    p = p->next;

    if (i > k + 1) {
      LOG(INFO) << "kp";
      kp = kp->next;
    }
    if (i > k + 2) {
      LOG(INFO) << "j";
      j = j->next;
    }
  }

  LOG(INFO) << "reached end";

  // Check if the list was long enough.
  if (i < k + 1) {
    return nullptr;
  }

  if (i > k + 1) {
    j->next = kp->next;
  } else {
    // k was the front of the list
    head = kp->next;
  }

  kp->next = nullptr;

  return head;
}

TEST(RemoveKLinkedList, EmptyList) { ASSERT_EQ(Remove(nullptr, 0), nullptr); }

TEST(RemoveKLinkedList, NegativeK) {
  L a;
  ASSERT_EQ(Remove(&a, -1), nullptr);
}

TEST(RemoveKLinkedList, LastElement) {
  L a, b, c;
  a.next = &b;
  b.next = &c;

  ASSERT_EQ(Remove(&a, 0), &a);
  ASSERT_EQ(a.next, &b);
  ASSERT_EQ(b.next, nullptr);
  ASSERT_EQ(c.next, nullptr);
}

TEST(RemoveKLinkedList, SecondToLastElement) {
  L a, b, c;
  a.next = &b;
  b.next = &c;

  ASSERT_EQ(Remove(&a, 1), &a);
  ASSERT_EQ(a.next, &c);
  ASSERT_EQ(b.next, nullptr);
  ASSERT_EQ(c.next, nullptr);
}

TEST(RemoveKLinkedList, FirstElement) {
  L a, b, c;
  a.next = &b;
  b.next = &c;

  ASSERT_EQ(Remove(&a, 2), &b);
  ASSERT_EQ(a.next, nullptr);
  ASSERT_EQ(b.next, &c);
  ASSERT_EQ(c.next, nullptr);
}

TEST(RemoveKLinkedList, OutOfRangeElement) {
  L a, b, c;
  a.next = &b;
  b.next = &c;

  ASSERT_EQ(Remove(&a, 3), nullptr);
}

TEST_MAIN();
