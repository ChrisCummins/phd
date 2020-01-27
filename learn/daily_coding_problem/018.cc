// This problem was asked by Snapchat.
//
// Given an array of time intervals (start, end) for classroom lectures
// (possibly overlapping), find the minimum number of rooms required.
//
// For example, given [(30, 75), (0, 50), (60, 150)], you should return 2.

#include <algorithm>
#include <utility>
#include <vector>

#include "labm8/cpp/test.h"

using std::pair;
using std::sort;
using std::vector;

int f(const vector<pair<int, int>>& in) {
  vector<int> rooms;

  vector<pair<int, int>> V{in.begin(), in.end()};
  sort(V.begin(), V.end());

  for (auto& room : V) {
    bool fr = false;
    for (size_t i = 0; i < rooms.size(); ++i) {
      if (rooms[i] < room.first) {
        rooms[i] = room.second;
        sort(rooms.begin(), rooms.end());
        fr = true;
        break;
      }
    }

    if (!fr) {
      rooms.push_back(room.second);
      sort(rooms.begin(), rooms.end());
    }
  }

  return rooms.size();
}

TEST(MinRoomsRequired, EmptyList) { EXPECT_EQ(f({}), 0); }

TEST(MinRoomsRequired, SingleRoomList) { EXPECT_EQ(f({{30, 75}}), 1); }

TEST(MinRoomsRequired, TwoRooms) { EXPECT_EQ(f({{30, 75}, {30, 75}}), 2); }

TEST(MinRoomsRequired, Example) {
  EXPECT_EQ(f({{30, 75}, {0, 50}, {60, 150}}), 2);
}

TEST(MinRoomsRequired, ExamplePermutation) {
  EXPECT_EQ(f({{60, 150}, {30, 75}, {0, 50}}), 2);
}

TEST_MAIN();
