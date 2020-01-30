// This problem was asked by Pinterest.
//
// At a party, there is a single person who everyone knows, but who
// does not know anyone in return (the "celebrity"). To help figure
// out who this is, you have access to an O(1) method called knows(a,
// b), which returns True if person a knows person b, else False.
//
// Given a list of N people and the above operation, find a way to
// identify the celebrity in O(N) time.
#include <stack>
#include <unordered_set>
#include <vector>
#include "labm8/cpp/test.h"

using std::stack;
using std::unordered_set;
using std::vector;

struct Person {
  unordered_set<const Person*> knows;
};

bool Knows(const Person& a, const Person& b) {
  return a.knows.find(&b) == a.knows.end() ? false : true;
}

// Time: O(n ^ 2)
// Space: O(1)
const Person* FindCelebAtPartyBruteForce(const vector<const Person*>& party) {
  for (int j = 0; j < party.size(); ++j) {
    bool celeb = true;
    for (int i = 0; i < party.size(); ++i) {
      if (j != i && Knows(*party[j], *party[i])) {
        celeb = false;
        break;
      }
    }
    if (celeb) {
      return party[j];
    }
  }
  return nullptr;
}

TEST(FindCelebAtParty, BruteForceEmptyList) {
  EXPECT_EQ(FindCelebAtPartyBruteForce({}), nullptr);
}

TEST(FindCelebAtParty, BruteForceExampleParty) {
  Person a, b, c, d, e;
  a.knows.insert(&b);
  c.knows.insert(&b);
  c.knows.insert(&d);
  d.knows.insert(&b);
  e.knows.insert(&b);

  EXPECT_EQ(FindCelebAtPartyBruteForce({&a, &b, &c, &d, &e}), &b);
}

const Person* FindCelebAtPartyStack(const vector<const Person*>& party) {
  stack<const Person*> s;
  for (const Person* person : party) {
    s.push(person);
  }

  while (s.size() > 1) {
    const Person* a = s.top();
    s.pop();
    const Person* b = s.top();
    s.pop();

    if (Knows(*a, *b)) {
      // a is not a celeb
      s.push(b);
    } else {
      s.push(a);
    }
  }

  if (s.empty()) {
    return nullptr;
  }

  return s.top();
}

TEST(FindCelebAtParty, EmptyList) {
  EXPECT_EQ(FindCelebAtPartyStack({}), nullptr);
}

TEST(FindCelebAtParty, ExampleParty) {
  Person a, b, c, d, e;
  a.knows.insert(&b);
  c.knows.insert(&b);
  c.knows.insert(&d);
  d.knows.insert(&b);
  e.knows.insert(&b);

  EXPECT_EQ(FindCelebAtPartyStack({&a, &b, &c, &d, &e}), &b);
}

TEST_MAIN();
