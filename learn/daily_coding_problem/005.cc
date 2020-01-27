// An XOR linked list is a more memory efficient doubly linked list. Instead of
// each node holding next and prev fields, it holds a field named both, which is
// an XOR of the next node and the previous node. Implement an XOR linked list;
// it has an add(element) which adds the element to the end, and a get(index)
// which returns the node at index.
//
// If using a language that has no pointers (such as Python), you can assume you
// have access to get_pointer and dereference_pointer functions that converts
// between nodes and memory addresses.

#include <cstdint>
#include <iostream>

class XorLinkedList {
 public:
  int value_;
  XorLinkedList* both_;

  XorLinkedList(const int& value) : value_(value), both_(nullptr) {}

  static XorLinkedList* Xor(const XorLinkedList* a, const XorLinkedList* b) {
    return (XorLinkedList*)((uintptr_t)(void*)a ^ (uintptr_t)(void*)b);
  }

  void add(XorLinkedList* element) {
    element->both_ = Xor(this, NULL);
    auto next = Xor(both_, NULL);
    both_ = Xor(element, next);
  }

  const XorLinkedList* get(const size_t n) {
    const XorLinkedList* curr = this;
    const XorLinkedList* prev = nullptr;

    for (int i = 0; i < n; ++i) {
      const XorLinkedList* next = XorLinkedList::Xor(prev, curr->both_);
      if (next == nullptr) {
        return nullptr;
      }
      prev = curr;
      curr = next;
    }

    return curr;
  }

  static void Print(const XorLinkedList* list) {
    const XorLinkedList* curr = list;
    const XorLinkedList* prev = nullptr;

    std::cout << "{ ";
    while (curr != nullptr) {
      std::cout << curr->value_ << ", ";

      const XorLinkedList* next = XorLinkedList::Xor(prev, curr->both_);
      prev = curr;
      curr = next;
    }
    std::cout << '}' << std::endl;
  }
};

int main(int argc, char** argv) {
  XorLinkedList A(1);
  XorLinkedList B(2);
  XorLinkedList C(3);

  A.add(&B);
  B.add(&C);

  //  XorLinkedList::Print(&A);
  //  XorLinkedList::Print(A.get(0));
  //  XorLinkedList::Print(A.get(1));
  //  XorLinkedList::Print(&B);
  //  XorLinkedList::Print(&C);

  return 0;
}
