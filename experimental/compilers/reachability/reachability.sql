-- How many CFGs do we have, and where have they come from?
SELECT `reachability`.`llvm_bytecode`.source_name AS source,COUNT(*) as count
FROM `reachability`.`control_flow_graph_proto`
LEFT JOIN `reachability`.`llvm_bytecode` ON bytecode_id=`reachability`.`llvm_bytecode`.id
WHERE status = 0
GROUP BY `reachability`.`llvm_bytecode`.source_name
ORDER BY count DESC;