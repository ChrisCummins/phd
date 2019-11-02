-- Aggregate stats about bytecode corpus.

SELECT graphs.source_name,
       graphs.language,
       file_count,
       count(*) AS graph_count,
       sum(linecount) AS line_count,
       sum(node_count) AS node_count,
       sum(edge_count) AS edge_count,
       avg(greatest(1, edge_count) / node_count) AS branch_factor
FROM ml4pl_unlabelled_corpus.graph_metas AS graphs
LEFT JOIN reachability.llvm_bytecode AS bytecode ON graphs.bytecode_id = bytecode.id
LEFT JOIN
  (SELECT source_name AS source_name_,
          count(*) AS file_count
   FROM reachability.llvm_bytecode
   WHERE clang_returncode=0
   GROUP BY source_name_) AS t ON source_name_=graphs.source_name
GROUP BY source_name,
         `language`;

-- Check for duplicates in bytecode sources.

SELECT bytecode_sha1,
       `count`
FROM
  (SELECT bytecode_sha1,
          count(*) AS 'count'
   FROM reachability.llvm_bytecode
   WHERE clang_returncode=0
   GROUP BY bytecode_sha1) AS t
ORDER BY `count` DESC;

-- Agregate stats about the graph database.

SELECT count(*)
FROM ml4pl_unlabelled_corpus.graph_metas;


SELECT source_name,
       `language`,
       `group`,
       count(*) AS graph_count,
       sum(node_count) AS node_count,
       sum(edge_count) AS edge_count,
       round(avg(node_count)) AS avg_node_count,
       round(avg(edge_count)) AS avg_edge_count
FROM ml4pl_unlabelled_corpus.graph_metas
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

