// An implementation of minimum spanning trees using Kruskal's algorithm.
#pragma once

#include "boost/graph/adjacency_list.hpp"

namespace phd {
namespace learn {

// Edges have integer weights.
using EdgeWeightProperty = boost::property<boost::edge_weight_t, int>;

// We're dealing with undirected graphs with weighted edges.
using Graph = boost::adjacency_list<
    /*OutEdgeList=*/boost::vecS, /*VertexList=*/boost::vecS,
    /*Directed=*/boost::undirectedS,
    /*VertexProperties=*/boost::no_property,
    /*EdgeProperties=*/EdgeWeightProperty>;

// Get the number of vertices in the graph.
size_t VertexCount(const Graph& graph);

// Get the number of edges in the graph.
size_t EdgeCount(const Graph& graph);

// Return true if the graph contains cycles.
bool HasCycle(const Graph& graph);

// Return if the edge exists in the graph, and the edge descriptor. This is not
// as efficient as it could be, requiring an O(n) scan through the edges for a
// match.
std::pair<bool, boost::graph_traits<Graph>::edge_descriptor> FindEdge(
    const Graph& graph, const int source_index, const int target_index);

// Create a minimum spanning tree from a graph and return it.
//
// The original graph is modified.
//
// Kruskal's algorithm to find the minimum cost spanning tree uses the greedy
// approach. This algorithm treats the graph as a forest and every node it has
// as an individual tree. A tree connects to another only and only if, it has
// the least cost among all available options and does not violate MST
// properties.
Graph KruskalMinimumSpanningTree(Graph* graph);

}  // namespace learn
}  // namespace phd
