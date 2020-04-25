// This problem was asked by Spotify.

// You have access to ranked lists of songs for various users. Each song is
// represented as an integer, and more preferred songs appear earlier in each
// list. For example, the list [4, 1, 7] indicates that a user likes song 4 the
// best, followed by songs 1 and 7.

// Given a set of these ranked lists, interleave them to create a playlist that
// satisfies everyone's priorities.

// For example, suppose your input is {[1, 7, 3], [2, 1, 6, 7, 9], [3, 9, 5]}.
// In this case a satisfactory playlist could be [2, 1, 6, 7, 3, 9, 5].

#include "labm8/cpp/logging.h"
#include "labm8/cpp/test.h"

#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using std::queue;
using std::stack;
using std::unordered_map;
using std::unordered_set;
using std::vector;

// Topological sort from a map of adjacency sets to a sorted vector of nodes.
// Normally you use a stack to push the sorted nodes to, but since we know we
// want to eventually return a vector, we use a playlist vector and position
// index.
void TopSort(const unordered_map<int, unordered_set<int>>& graph, int song,
             unordered_set<int>* visited, vector<int>* playlist,
             int* position) {
  visited->insert(song);

  auto children = graph.find(song);
  CHECK(children != graph.end()) << "Could not find children for " << song;
  for (auto child : children->second) {
    if (visited->find(child) == visited->end()) {
      TopSort(graph, child, visited, playlist, position);
    }
  }

  if (song) {  // ignore song "0" - the root song.
    (*playlist)[*position] = song;
    --(*position);
  }
}

// Time: O(n)
// Space: O(n)
vector<int> MakePlaylist(const vector<vector<int>>& playlists) {
  // Build a graph of songs and their dependencies.
  unordered_map<int, unordered_set<int>> graph;
  graph.insert({0, {}});

  for (const auto& playlist : playlists) {
    // Add root entry.
    graph.find(0)->second.insert(playlist[0]);

    for (size_t i = 0; i < playlist.size(); ++i) {
      // Create the entry in the adjacencies map, even if we are not going to
      // add an adjacency.
      auto& children = graph[playlist[i]];
      if (i < playlist.size() - 1) {
        children.insert(playlist[i + 1]);
      }
    }
  }

  // Post-order DFS traversal to produce topological sorting of graph.
  vector<int> playlist(graph.size() - 1);
  int position = playlist.size() - 1;
  unordered_set<int> visited;

  TopSort(graph, 0, &visited, &playlist, &position);
  return playlist;
}

TEST(CombinedPlaylist, Empty) {
  vector<vector<int>> playlists({});

  auto playlist = MakePlaylist(playlists);
  EXPECT_EQ(playlist.size(), 0);
}

TEST(CombinedPlaylist, SingleInput) {
  vector<vector<int>> playlists({
      {1, 2, 3, 4, 5, 6},
  });

  auto playlist = MakePlaylist(playlists);
  ASSERT_EQ(playlist.size(), 6);

  EXPECT_EQ(playlist[0], 1);
  EXPECT_EQ(playlist[1], 2);
  EXPECT_EQ(playlist[2], 3);
  EXPECT_EQ(playlist[3], 4);
  EXPECT_EQ(playlist[4], 5);
  EXPECT_EQ(playlist[5], 6);
}

TEST(CombinedPlaylist, ExampleInput) {
  vector<vector<int>> playlists({{1, 7, 3}, {2, 1, 6, 7, 9}, {3, 9, 5}});

  auto playlist = MakePlaylist(playlists);
  ASSERT_EQ(playlist.size(), 7);

  EXPECT_EQ(playlist[0], 2);
  EXPECT_EQ(playlist[1], 1);
  EXPECT_EQ(playlist[2], 6);
  EXPECT_EQ(playlist[3], 7);
  EXPECT_EQ(playlist[4], 3);
  EXPECT_EQ(playlist[5], 9);
  EXPECT_EQ(playlist[6], 5);
}

TEST_MAIN();
