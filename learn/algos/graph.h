#ifndef GRAPH_H
#define GRAPH_H

#include <cassert>
#include <iostream>

template <typename T>
class Graph {
 public:
  Graph* left;
  Graph* right;
  T val;

  Graph(const T& _val) : val(_val) {}
};

void test_search(Graph<int>* graph, const int& val,
                 Graph<int>* search(Graph<int>*, const int&)) {
  assert(graph);

  std::cout << "traversal for " << val << ":";
  Graph<int>* ret = search(graph, val);
  std::cout << std::endl;
  if (ret) {
    std::cout << "found " << ret->val;
  } else {
    std::cout << "not found";
  }
  std::cout << std::endl << std::endl;
}

#endif  // GRAPH_H
