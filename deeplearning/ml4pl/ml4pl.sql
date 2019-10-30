-- Aggregate stats about bytecode corpus.

SELECT source_name,
       LANGUAGE,
       file_count,
       count(*) AS function_count,
       sum(linecount) AS line_count,
       sum(block_count) AS basic_block_count,
       avg(greatest(1, edge_count) / block_count) AS cfg_branch_factor
FROM reachability.control_flow_graph_proto AS cfg
LEFT JOIN reachability.llvm_bytecode AS bytecode ON cfg.bytecode_id = bytecode.id
LEFT JOIN
  (SELECT source_name AS source_name_,
          count(*) AS file_count
   FROM reachability.llvm_bytecode
   GROUP BY source_name) AS t ON source_name_=source_name
WHERE status=0
GROUP BY source_name,
         LANGUAGE;

-- Agregate stats about the graph database.

SELECT count(*)
FROM ml4pl_graphs.graph_metas;


SELECT source_name,
       `language`,
       `group`,
       count(*) AS graph_count,
       sum(node_count) AS node_count,
       sum(edge_count) AS edge_count,
       round(avg(node_count)) AS avg_node_count,
       round(avg(edge_count)) AS avg_edge_count
FROM ml4pl_graphs.graph_metas
GROUP BY source_name,
         `language`,
         `group`
ORDER BY source_name,
         `language`,
         `group`;

-- Compare file and graph counts, indicating how many failed.

SELECT graphs.source_name,
       graph_count,
       bytecode_count,
       bytecode_count - graph_count AS 'failed',
       (graph_count / bytecode_count) * 100 AS 'ratio'
FROM
  (SELECT source_name,
          count(*) AS graph_count
   FROM ml4pl_graphs.graph_metas
   GROUP BY source_name) AS graphs
LEFT JOIN
  (SELECT source_name,
          count(*) AS bytecode_count
   FROM reachability.llvm_bytecode
   GROUP BY source_name) AS bytecodes ON graphs.source_name = bytecodes.source_name
ORDER BY ratio ASC;

-- How many CFGs do we have, and where have they come from?

SELECT `reachability`.`llvm_bytecode`.source_name AS SOURCE,
       count(*) AS COUNT
FROM `reachability`.`control_flow_graph_proto`
LEFT JOIN `reachability`.`llvm_bytecode` ON bytecode_id=`reachability`.`llvm_bytecode`.id
WHERE status = 0
GROUP BY `reachability`.`llvm_bytecode`.source_name
ORDER BY COUNT DESC;

-- Labelled graphs database overview.

SELECT graphs.source_name AS SOURCE,
       graphs.language,
       graphs.group,
       count(*) AS COUNT
FROM ml4pl_reachability_cfg_all.graph_metas AS graphs
GROUP BY graphs.source_name,
         graphs.group,
         graphs.language
ORDER BY graphs.group,
         COUNT DESC;


SELECT graphs.language,
       graphs.group,
       count(*) AS COUNT
FROM ml4pl_reachability_cfg_all.graph_metas AS graphs
GROUP BY graphs.group,
         graphs.language
ORDER BY graphs.group,
         COUNT DESC;

