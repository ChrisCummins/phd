"""Convert a SQL workspace to dot source."""
import json
from argparse import ArgumentParser
from typing import List

from graphviz import Digraph

from util.freefocus.sql import *


def build_graph(
  session,
  make_id: "lambda item: object, type: str",
  add_node: "lambda id: str, item: object, type: str",
  add_edge: "lambda from_id: str, to_id: str, type: Tuple[str, str]",
):
  def plot_items(parent_label, items: List[Base], node_type, edge_type):
    for item in items:
      label = make_id(item, type=node_type)
      if not add_node(label, item, type=node_type):
        continue
      add_edge(parent_label, label, type=edge_type)

      if hasattr(item, "created_by") and item.created_by:
        add_edge(
          make_id(item.created_by, type="group"),
          label,
          type=("person", edge_type[1]),
        )

      if hasattr(item, "members"):
        plot_items(
          label, item.members, node_type="person", edge_type=("group", "person")
        )

      if hasattr(item, "comments"):
        plot_items(
          label,
          item.comments,
          node_type="comment",
          edge_type=(edge_type[1], "comment"),
        )

      if hasattr(item, "tags"):
        for tag in item.tags:
          add_edge(make_id(tag, type="tag"), label, type=("tag", edge_type[1]))

      if hasattr(item, "assets"):
        for asset in item.assets:
          add_edge(
            make_id(asset, type="asset"), label, type=("asset", edge_type[1])
          )

      if hasattr(item, "children"):
        plot_items(
          label, item.children, node_type, (edge_type[1], edge_type[1])
        )

  workspace = session.query(Workspace).one()
  workspace_label = make_id(workspace, "workspace")
  add_node(workspace_label, workspace, type="workspace")

  plot_items(  # groups
    workspace_label,
    session.query(Group).filter(Group.parent == None).all(),
    node_type="group",
    edge_type=("workspace", "group"),
  )

  plot_items(  # assets
    workspace_label,
    session.query(Asset).filter(Asset.parent == None).all(),
    node_type="asset",
    edge_type=("workspace", "asset"),
  )

  plot_items(  # tags
    workspace_label,
    session.query(Tag).filter(Tag.parent == None).all(),
    node_type="tag",
    edge_type=("workspace", "tag"),
  )

  plot_items(  # tasks
    workspace_label,
    session.query(Task).filter(Task.parent == None).all(),
    node_type="task",
    edge_type=("workspace", "task"),
  )


if __name__ == "__main__":
  parser = ArgumentParser(description=__doc__)
  parser.add_argument("uri")
  parser.add_argument("-v", "--verbose", action="store_true")
  parser.add_argument("--html", action="store_true")
  parser.add_argument("--complete", action="store_true")
  parser.add_argument("--created-by", action="store_true")
  parser.add_argument("--tag-refs", action="store_true")
  parser.add_argument("--asset-refs", action="store_true")
  args = parser.parse_args()

  engine = sql.create_engine(args.uri, echo=args.verbose)

  Base.metadata.bind = engine
  make_session = sql.orm.sessionmaker(bind=engine)

  session = make_session()

  def build_html():
    graph = {"nodes": [], "links": []}

    def escape(string):
      line = str(string).strip().split("\n")[0]
      if len(line) > 30:
        return line[:27] + "..."
      else:
        return line

    def make_id(item, type):
      if type == "workspace":
        return escape(f"Workspace: {item.uid}")
      elif type == "person":
        return escape(f"Person: {item.name}")
      elif type == "group":
        return escape(f"Group: {item.body}")
      elif type == "asset":
        return escape(f"Asset: {item.body}")
      elif type == "tag":
        return escape(f"Tag: {item.body}")
      elif type == "task":
        return escape(f"Task: {item.body}")
      elif type == "comment":
        return escape(f"Comment: {item.body}")
      else:
        raise LookupError(type)

    def add_node(id, item, type):
      if hasattr(item, "completed"):
        if item.completed != None and not args.complete:
          return False

      groups = {
        "workspace": 1,
        "person": 2,
        "group": 3,
        "asset": 4,
        "tag": 5,
        "task": 6,
        "comment": 7,
      }

      graph["nodes"].append({"id": id, "group": groups[type]})
      return True

    def add_edge(from_id, to_id, type):
      values = {
        "workspace": 7,
        "person": 6,
        "group": 5,
        "asset": 4,
        "tag": 3,
        "task": 2,
        "comment": 1,
      }

      if type[0] == "person" and not args.created_by:
        return
      if type[0] == "tag" and type[1] == "task" and not args.tag_refs:
        return
      if type[0] == "asset" and type[1] == "task" and not args.asset_refs:
        return

      graph["links"].append(
        {"source": from_id, "target": to_id, "value": values[type[1]]}
      )

    build_graph(session, make_id, add_node, add_edge)

    print(
      """\
<!DOCTYPE html>
<meta charset="utf-8">
<!-- https://bl.ocks.org/mbostock/4062045 -->

<style>
.links line {
  stroke: #999;
  stroke-opacity: 0.6;
}

.nodes circle {
  stroke: #fff;
  stroke-width: 1.5px;
}
</style>

<svg width="960" height="600"></svg>
<script src="https://d3js.org/d3.v4.min.js"></script>

<script>
var svg = d3.select("svg"),
    width = +svg.attr("width"),
    height = +svg.attr("height");


function redraw() {
  svg.selectAll("*").remove();  /* reset svg */

  var color = d3.scaleOrdinal(d3.schemeCategory20);

  var simulation = d3.forceSimulation()
      .force("link", d3.forceLink().id(function(d) { return d.id; }))
      .force("charge", d3.forceManyBody())
      .force("center", d3.forceCenter(width / 2, height / 2));

  graph = %s;

  var link = svg.append("g").attr("class", "links")
    .selectAll("line")
    .data(graph.links)
    .enter().append("line")
      .attr("stroke-width", function(d) { return Math.sqrt(d.value); });

  var node = svg.append("g").attr("class", "nodes")
    .selectAll("circle")
    .data(graph.nodes)
    .enter().append("circle")
      .attr("r", 5)
      .attr("fill", function(d) { return color(d.group); })
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended));

  node.append("title")
      .text(function(d) { return d.id; });

  simulation
      .nodes(graph.nodes)
      .on("tick", ticked);

  simulation.force("link")
      .links(graph.links);

  function ticked() {
    link
        .attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    node
        .attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });
  }

  function dragstarted(d) {
    if (!d3.event.active)
      simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(d) {
    d.fx = d3.event.x;
    d.fy = d3.event.y;
  }

  function dragended(d) {
    if (!d3.event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }

  console.log("REDRAW\\n");
}

redraw();
</script>

<button onclick="redraw()">Redraw</button>
"""
      % json.dumps(graph, indent=2, separators=(",", ": "))
    )

  def build_dot():
    dot = Digraph(comment="FreeFocus")
    dot.graph_attr["rankdir"] = "LR"

    def escape(string):
      line = str(string).strip().split("\n")[0].replace(":", "/")
      if len(line) > 30:
        return line[:27] + "..."
      else:
        return line

    def make_id(item, type):
      if type == "workspace":
        return escape(item.uid)
      elif type == "person":
        return escape(f"__persons__.{item.name}")
      elif type == "group":
        return escape(f"__groups__.{item.body}")
      elif type == "asset":
        return escape(f"__assets__.{item.body}")
      elif type == "tag":
        return escape(f"__tags__.{item.body}")
      elif type == "task":
        return escape(f"__tasks__.{item.body}")
      elif type == "comment":
        return escape(f"__comments__.{item.body}")
      else:
        raise LookupError(type)

    def add_node(id, item, type):
      if hasattr(item, "completed"):
        if item.completed != None and not args.complete:
          return False

      node_opts = {"shape": "rect", "style": "filled"}

      if type == "workspace":
        name = item.uid
      elif type == "person":
        name = f"@persons//{item.name}"
        node_opts["fillcolor"] = "#FFD1DC"
      elif type == "group":
        name = f"@groups//{item.body}"
        node_opts["fillcolor"] = "#77DD77"
      elif type == "asset":
        name = f"@assets//{item.body}"
        node_opts["fillcolor"] = "#FFD1DC"
      elif type == "tag":
        name = f"@tags//{item.body}"
        node_opts["fillcolor"] = "#AEC6CF"
      elif type == "task":
        name = f"@tasks//{item.body}"
        node_opts["fillcolor"] = "#FFB347"
      elif type == "comment":
        name = f"@comments//{item.body}"
        node_opts["fillcolor"] = "#FDFD96"
      else:
        name = id

      dot.node(escape(id), escape(name), **node_opts)
      return True

    def add_edge(from_id, to_id, type):
      if type[0] == "person" and not args.created_by:
        return
      if type[0] == "tag" and type[1] == "task" and not args.tag_refs:
        return
      if type[0] == "asset" and type[1] == "task" and not args.asset_refs:
        return

      dot.edge(escape(from_id), escape(to_id))

    build_graph(session, make_id, add_node, add_edge)

    print(dot.source)

  if args.html:
    build_html()
  else:
    build_dot()
