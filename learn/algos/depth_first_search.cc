#include "./graph.h"

#include <cassert>
#include <set>

/*
 * Accepts a graph and a value to search for, and returns a pointer to the Graph
 * within the graph matching the value.
 *
 * Returns nullptr if value is not found in graph.
 */
template <typename T>
Graph<T>* _dfs(Graph<T>* graph, const T& val,
               std::set<Graph<T>*> visited = std::set<Graph<T>*>()) {
  assert(graph);
  std::cout << " " << graph->val;

  if (graph->val == val) return graph;

  visited.insert(graph);

  if (graph->left && visited.find(graph->left) == visited.end())
    return _dfs(graph->left, val, visited);

  if (graph->right && visited.find(graph->right) == visited.end())
    return _dfs(graph->right, val, visited);

  return nullptr;
}

template <typename T>
Graph<T>* dfs(Graph<T>* graph, const T& val) {
  return _dfs<T>(graph, val);
}

int main() {
  // assemble example graph
  // from http://www.geeksforgeeks.org/breadth-first-traversal-for-a-graph/
  Graph<int> graph(2);
  Graph<int> g0(0), g1(1), g2(2), g3(3);

  g0.left = &g1;
  g0.right = &graph;
  graph.left = &g0;
  graph.right = &g3;
  g3.left = &g3;
  g1.left = &graph;

  test_search(&graph, 2, dfs);
  test_search(&graph, 0, dfs);
  test_search(&graph, 1, dfs);
  // FIXME: test_search(&graph, 3, dfs);
  // FIXME: test_search(&graph, -1, dfs);
}
