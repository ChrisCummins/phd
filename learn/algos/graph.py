#!/usr/bin/env python3
from collections import deque
from typing import List


class Graph(object):
  """ adjacency lists graphs """

  def __init__(self):
    self.adjacencies = []

  def add_vertex(self):
    self.adjacencies.append([])
    return len(self) - 1

  def add_edge(self, start, end):
    if not (start in self and end in self):
      raise ValueError(start, end)

    self.adjacencies[start].append(end)

  def add_edges(self, *edges):
    for edge in edges:
      self.add_edge(*edge)

  def get_adjacent_vertices(self, vertex):
    """ return neighboring vertices """
    if not vertex in self:
      raise ValueError
    return self.adjacencies[vertex]

  def __len__(self):
    """ get number of vertices """
    return len(self.adjacencies)

  @property
  def vertices(self):
    """ iterate over vertices """
    return range(len(self.adjacencies))

  @property
  def edges(self):
    """ iterate over (x, y) graph edges """
    for vertex in self.vertices:
      for adj in self.get_adjacent_vertices(vertex):
        yield (vertex, adj)

  def __repr__(self):
    return """\
V = ({})
E = (
  {}
)
""".format(
      ", ".join([str(x) for x in self.vertices]),
      ",\n  ".join([f"({x[0]} -> {x[1]})" for x in self.edges]),
    )

  def __contains__(self, vertex: int):
    return vertex >= 0 and vertex < len(self)

  def shortest_path(self, source, dest) -> List[int]:
    """ fetch shortest path, or None if not found """
    if source not in self or dest not in self:
      raise ValueError

    # breadth first search, where the queue is a queue of paths, rather than
    # nodes.
    #
    # TODO: Modify to Dijkstra's algorithm, which uses less memory.
    visited = [False] * len(self.vertices)

    q = deque([[source]])
    visited[source] = True

    while len(q):
      path = q.popleft()
      vertex = path[-1]

      if vertex == dest:
        return path

      for neighbor in self.get_adjacent_vertices(vertex):
        if not visited[neighbor]:
          q.append(path + [neighbor])
          visited[neighbor] = True

    return None


if __name__ == "__main__":
  g = Graph()
  v0 = g.add_vertex()
  v1 = g.add_vertex()
  v2 = g.add_vertex()
  v3 = g.add_vertex()
  v4 = g.add_vertex()
  v5 = g.add_vertex()

  g.add_edges(
    (v0, v1),
    (v0, v2),
    (v1, v2),
    (v1, v3),
    (v2, v4),
    (v3, v4),
    (v4, v5),
    (v3, v5),
  )

  print(g)
  assert g.shortest_path(0, 4) == [0, 2, 4]
  assert g.shortest_path(0, 0) == [0]
  assert g.shortest_path(0, 5) == [0, 1, 3, 5]
  assert g.shortest_path(5, 0) == None
