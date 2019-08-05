#include "learn/daily/d181213_kruskal_spanning_tree.h"

#include "labm8/cpp/logging.h"

#include "absl/container/flat_hash_map.h"

#include <boost/graph/depth_first_search.hpp>

#include <set>

namespace labm8 {
namespace learn {

size_t VertexCount(const Graph& graph) {
  boost::graph_traits<Graph>::vertex_iterator i;
  boost::graph_traits<Graph>::vertex_iterator end;
  boost::tie(i, end) = boost::vertices(graph);
  return std::distance(i, end);
}

size_t EdgeCount(const Graph& graph) {
  boost::graph_traits<Graph>::edge_iterator i;
  boost::graph_traits<Graph>::edge_iterator end;
  boost::tie(i, end) = boost::edges(graph);
  return std::distance(i, end);
}

bool HasCycle(const Graph& graph,
              const boost::property_map<Graph, boost::vertex_index_t>::type&
                  vertex_index_map,
              boost::graph_traits<Graph>::vertex_descriptor vertex,
              std::set<unsigned long>* visited,
              std::set<unsigned long>* recursion_stack) {
  boost::graph_traits<Graph>::adjacency_iterator i;
  boost::graph_traits<Graph>::adjacency_iterator end;

  auto vertex_index = vertex_index_map[vertex];
  if (visited->find(vertex_index) == visited->end()) {
    LOG(DEBUG) << "Visiting new vertex " << vertex_index;
    // Mark the current node as visited and part of recursion stack.
    visited->insert(vertex_index);
    recursion_stack->insert(vertex_index);

    // Recur for all the vertices adjacent to this vertex.
    for (boost::tie(i, end) = boost::adjacent_vertices(vertex, graph); i != end;
         ++i) {
      // FIXME(cec): It seems that the adjacency iterator cannot be converted
      // to a vertex index using this approach. I think the integer value is
      // an index into the ajacency list, not the vertex list. This causes test
      // failure:
      // [ RUN      ] HasCycle.CycleFreeGraph
      // D 2018-12-14 17:46:08 [learn/daily/d181213_kruskal_spanning_tree.cc:39]
      // Visiting new vertex 0 D 2018-12-14 17:46:08
      // [learn/daily/d181213_kruskal_spanning_tree.cc:48] Visiting adjacent
      // vertex 0 -> 1 D 2018-12-14 17:46:08
      // [learn/daily/d181213_kruskal_spanning_tree.cc:39] Visiting new vertex 1
      // D 2018-12-14 17:46:08 [learn/daily/d181213_kruskal_spanning_tree.cc:48]
      // Visiting adjacent vertex 1 -> 0 D 2018-12-14 17:46:08
      // [learn/daily/d181213_kruskal_spanning_tree.cc:56] Adjacent vertex 1 ->
      // 0 in recursion stack D 2018-12-14 17:46:08
      // [learn/daily/d181213_kruskal_spanning_tree.cc:52] Adjacent vertex 0 ->
      // 1 has cycle
      unsigned long adjacent_index = vertex_index_map[*i];
      LOG(DEBUG) << "Visiting adjacent vertex " << vertex_index << " -> "
                 << adjacent_index;
      if (visited->find(adjacent_index) == visited->end() &&
          HasCycle(graph, vertex_index_map, *i, visited, recursion_stack)) {
        LOG(DEBUG) << "Adjacent vertex " << vertex_index << " -> "
                   << adjacent_index;
        return true;
      } else if (recursion_stack->find(adjacent_index) !=
                 recursion_stack->end()) {
        LOG(DEBUG) << "Adjacent vertex " << vertex_index << " -> "
                   << adjacent_index << " in recursion stack";
        return true;
      }
    }
  }
  // Remove the current node from recursion stack.
  recursion_stack->erase(vertex_index);
  return false;
}

bool HasCycle(const Graph& graph) {
  std::set<unsigned long> visited;
  std::set<unsigned long> recursion_stack;

  boost::graph_traits<Graph>::vertex_iterator i;
  boost::graph_traits<Graph>::vertex_iterator end;
  boost::tie(i, end) = boost::vertices(graph);

  if (i != end) {
    auto vertex_index_map = get(boost::vertex_index, graph);
    return HasCycle(graph, vertex_index_map, *i, &visited, &recursion_stack);
  } else {
    return false;
  }
}

std::pair<bool, boost::graph_traits<Graph>::edge_descriptor> FindEdge(
    const Graph& graph, const int source_index, const int target_index) {
  boost::graph_traits<Graph>::edge_iterator i;
  boost::graph_traits<Graph>::edge_iterator end;

  for (boost::tie(i, end) = boost::edges(graph); i != end; ++i) {
    int edge_source = boost::source(*i, graph);
    int edge_target = boost::target(*i, graph);

    if (edge_source == source_index && edge_target == target_index) {
      return std::make_tuple(true, *i);
    }
  }
  return std::make_tuple(false, *end);
}

Graph KruskalMinimumSpanningTree(Graph* graph) {
  // Edge iterators.
  boost::graph_traits<Graph>::edge_iterator i;
  boost::graph_traits<Graph>::edge_iterator end;

  // Step 1 - Remove all loops and Parallel Edges.

  // A map of edges which is used to detect parallel edges. The key is a
  // <source, target> index tuple for the edge, and the value is a
  // <weight, edge_iterator> pair. Every time we visit a new edge, we check to
  // see if the source and target are in the map. If they are not, we insert
  // a new entry with the weight and iterator of the current edge. Else, we
  // compare the weight of the value in the map. If the visited weight is less
  // than the current edge weight, we remove the current edge. Else, we remove
  // the edge stored in the map, and replace the entry with the current edge.
  absl::flat_hash_map<std::pair<int, int>,
                      std::pair<int, boost::graph_traits<Graph>::edge_iterator>>
      visited_edges_map;

  for (boost::tie(i, end) = boost::edges(*graph); i != end; ++i) {
    int source_index = boost::source(*i, *graph);
    int target_index = boost::target(*i, *graph);

    // Remove self loop.
    if (source_index == target_index) {
      LOG(DEBUG) << "Removing self loop " << source_index << " -> "
                 << target_index;
      boost::remove_edge(*i, *graph);
      continue;
    }

    int weight = get(boost::edge_weight, *graph, *i);
    decltype(visited_edges_map)::key_type map_key(source_index, target_index);
    decltype(visited_edges_map)::mapped_type map_value(weight, i);
    LOG(DEBUG) << "Visiting edge " << source_index << " -> " << target_index
               << " with weight " << weight;

    auto map_it = visited_edges_map.find(map_key);
    if (map_it == visited_edges_map.end()) {
      visited_edges_map.insert(std::make_pair(map_key, map_value));
    } else {
      int visited_weight = map_it->second.first;
      boost::graph_traits<Graph>::edge_iterator visited_edge =
          map_it->second.second;
      if (weight < visited_weight) {
        LOG(DEBUG) << "Removing previous parallel edge " << source_index
                   << " -> " << target_index << " with weight "
                   << visited_weight << " (new parallel edge has lower weight "
                   << weight << ")";
        boost::remove_edge(*visited_edge, *graph);
        visited_edges_map.insert(std::make_pair(map_key, map_value));
      } else {
        LOG(DEBUG) << "Removing parallel edge " << source_index << " -> "
                   << target_index << " with weight " << weight
                   << " (already visited parallel edge with weight "
                   << visited_weight << ")";
        boost::remove_edge(*i, *graph);
      }
    }
  }

  // Step 2 - Arrange all edges in their increasing order of weight.

  // Create a vector of <weight, edge_iterator> pairs, and sort it by ascending
  // weight.
  boost::tie(i, end) = boost::edges(*graph);
  std::vector<std::pair<int, boost::graph_traits<Graph>::edge_iterator>>
      sorted_edges;
  sorted_edges.reserve(std::distance(i, end));
  for (; i != end; ++i) {
    int weight = get(boost::edge_weight, *graph, *i);
    sorted_edges.push_back(std::make_pair(weight, i));
  }
  std::sort(
      sorted_edges.begin(), sorted_edges.end(),
      // Sort the elements by ascending weights.
      [](const std::pair<int, boost::graph_traits<Graph>::edge_iterator>& a,
         const std::pair<int, boost::graph_traits<Graph>::edge_iterator>& b)
          -> bool { return a.first < b.first; });

  // Step 3 - Add the edge which has the least weightage.

  Graph mst(VertexCount(*graph));

  for (auto element : sorted_edges) {
    int weight = element.first;
    boost::graph_traits<Graph>::edge_iterator edge = element.second;
    LOG(DEBUG) << "Trying to add edge with weight " << weight;

    int source_index = boost::source(*edge, *graph);
    int target_index = boost::target(*edge, *graph);

    std::pair<boost::graph_traits<Graph>::edge_descriptor, bool> new_edge =
        boost::add_edge(source_index, target_index, weight, mst);
    CHECK(new_edge.second);
    LOG(DEBUG) << "Added MST edge " << source_index << " -> " << target_index;

    // If the new edge introduces a cycle, it breaks the MST properties. Remove
    // it and move onto the next edge.
    if (HasCycle(mst)) {
      LOG(DEBUG) << "Edge " << source_index << " -> " << target_index
                 << " introduced cycle, removing";
      boost::remove_edge(new_edge.first, mst);
    }
  }

  return mst;
}

}  // namespace learn
}  // namespace labm8
