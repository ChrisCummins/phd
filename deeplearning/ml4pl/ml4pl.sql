-- How many bytecodes do we have, and where have they come from?

SELECT `reachability`.`llvm_bytecode`.source_name AS SOURCE,
       count(*) AS COUNT
FROM `reachability`.`llvm_bytecode`
WHERE clang_returncode = 0
GROUP BY `reachability`.`llvm_bytecode`.source_name
ORDER BY COUNT DESC;

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

