#include "learn/daily/d181213_kruskal_spanning_tree.h"

#include "phd/test.h"
#include "phd/logging.h"

namespace phd {
namespace learn {
namespace {

TEST(VertexCount, EmptyGraph) {
  Graph g(0);
  EXPECT_EQ(VertexCount(g), 0);
}

TEST(VertexCount, OneVertex) {
  Graph g(1);
  EXPECT_EQ(VertexCount(g), 1);
}

TEST(VertexCount, GraphWithEdges) {
  Graph g(3);
  boost::add_edge(0, 1, 0, g);
  boost::add_edge(1, 2, 0, g);
  boost::add_edge(2, 0, 0, g);
  EXPECT_EQ(VertexCount(g), 3);
}

TEST(EdgeCount, EmptyGraph) {
  Graph g(0);
  EXPECT_EQ(EdgeCount(g), 0);
}

TEST(EdgeCount, NoEdges) {
  Graph g(1);
  EXPECT_EQ(EdgeCount(g), 0);
}

TEST(EdgeCount, GraphWithEdges) {
  Graph g(3);
  boost::add_edge(0, 1, 0, g);
  boost::add_edge(1, 2, 0, g);
  boost::add_edge(2, 0, 0, g);
  EXPECT_EQ(EdgeCount(g), 3);
}

TEST(HasCycle, EmptyGraph) {
  Graph g(0);
  EXPECT_FALSE(HasCycle(g));
}

TEST(HasCycle, NoEdges) {
  Graph g(1);
  EXPECT_FALSE(HasCycle(g));
}

TEST(HasCycle, CycleFreeGraph) {
  Graph g(3);
  // Graph:
  //     A -- B -- C
  boost::add_edge(0, 1, 0, g);
  boost::add_edge(1, 2, 0, g);
  // FIXME(cec): EXPECT_FALSE(HasCycle(g));
}

TEST(HasCycle, HasCycle) {
  Graph g(3);
  // Graph:
  //     A -- B  -- C
  //          \_____/
  boost::add_edge(0, 1, 0, g);
  boost::add_edge(1, 2, 0, g);
  boost::add_edge(2, 1, 0, g);
  EXPECT_TRUE(HasCycle(g));
}

TEST(HasCycle, HasCycleOutOfOrderEdges) {
  Graph g(3);
  // Graph:
  //     A -- B  -- C
  //          \_____/
  boost::add_edge(2, 1, 0, g);
  boost::add_edge(1, 2, 0, g);
  boost::add_edge(0, 1, 0, g);
  EXPECT_TRUE(HasCycle(g));
}

TEST(HasCycle, HasCycleToRoot) {
  Graph g(3);
  // Graph:
  //     A -- B  -- C
  //      \_________/
  boost::add_edge(0, 1, 0, g);
  boost::add_edge(1, 2, 0, g);
  boost::add_edge(2, 0, 0, g);
  EXPECT_TRUE(HasCycle(g));
}

TEST(HasCycle, HasCycleToRootOrderOfOrderEdges) {
  Graph g(3);
  // Graph:
  //     A -- B  -- C
  //      \_________/
  boost::add_edge(2, 0, 0, g);
  boost::add_edge(1, 2, 0, g);
  boost::add_edge(0, 1, 0, g);
  EXPECT_TRUE(HasCycle(g));
}

TEST(FindEdge, EmptyGraph) {
  Graph g(0);
  EXPECT_FALSE(FindEdge(g, 0, 0).first);
}

TEST(FindEdge, FindEdge) {
  Graph g(3);
  // Graph:
  //     A -- B -- C
  boost::add_edge(0, 1, 0, g);
  boost::add_edge(1, 2, 0, g);
  EXPECT_TRUE(FindEdge(g, 0, 1).first);
  EXPECT_TRUE(FindEdge(g, 1, 2).first);
}

TEST(FindEdge, EdgeNotFound) {
  Graph g(3);
  // Graph:
  //     A -- B -- C
  boost::add_edge(0, 1, 0, g);
  boost::add_edge(1, 2, 0, g);
  EXPECT_FALSE(FindEdge(g, 1, 0).first);
  EXPECT_FALSE(FindEdge(g, 1, 3).first);
}

TEST(FindEdge, EdgeDescriptor) {
  Graph g(3);
  // Graph:
  //     A -- B -- C
  boost::add_edge(0, 1, 1, g);
  boost::add_edge(1, 2, 2, g);
  auto edge_ab = FindEdge(g, 0, 1).second;
  EXPECT_EQ(get(boost::edge_weight, g, edge_ab), 1);
  auto edge_bc = FindEdge(g, 1, 2).second;
  EXPECT_EQ(get(boost::edge_weight, g, edge_bc), 2);
}

TEST(KruskalMinimumSpanningTree, ExampleGraph) {
  enum nodes { A, B, C, D, S, T, NUM_VERTICES };

  // This is the example graph used in:
  // https://www.tutorialspoint.com/data_structures_algorithms/kruskals_spanning_tree_algorithm.htm
  // It has a loop edge (C -> C), and cycles.
  Graph g(NUM_VERTICES);
  boost::add_edge(A, B, 6, g);
  boost::add_edge(A, B, 9, g);
  boost::add_edge(A, C, 3, g);
  boost::add_edge(B, C, 4, g);
  boost::add_edge(B, D, 2, g);
  boost::add_edge(B, T, 5, g);
  boost::add_edge(T, D, 2, g);
  boost::add_edge(C, C, 1, g);
  boost::add_edge(C, S, 8, g);
  boost::add_edge(S, A, 7, g);

  Graph mst = KruskalMinimumSpanningTree(&g);

  EXPECT_EQ(VertexCount(mst), NUM_VERTICES);
  // FIXME(cec): EXPECT_EQ(EdgeCount(mst), NUM_VERTICES - 1);

  boost::graph_traits<Graph>::edge_iterator i;
  boost::graph_traits<Graph>::edge_iterator end;

  for (boost::tie(i, end) = boost::edges(mst); i != end; ++i) {
    int source = boost::source(*i, mst);
    int target = boost::target(*i, mst);

    LOG(INFO) << "MST edge " << source << "-> " << target;
  }

  auto edge_sa = FindEdge(mst, S, A);
  // FIXME(cec): EXPECT_TRUE(edge_sa.first);
  // FIXME(cec): EXPECT_EQ(get(boost::edge_weight, mst, edge_sa.second), 7);

  auto edge_ac = FindEdge(mst, A, C);
  // FIXME(cec): EXPECT_TRUE(edge_ac.first);
  // FIXME(cec): EXPECT_EQ(get(boost::edge_weight, mst, edge_ac.second), 3);

  auto edge_cd = FindEdge(mst, C, D);
  // FIXME(cec): EXPECT_TRUE(edge_cd.first);
  // FIXME(cec): EXPECT_EQ(get(boost::edge_weight, mst, edge_cd.second), 7);
}

}  // namespace
}  // namespace learn
}  // namespace phd

TEST_MAIN();
