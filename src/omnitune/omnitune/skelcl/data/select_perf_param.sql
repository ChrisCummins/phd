-- Return the performance of each scenario relative to the oracle for
-- a given param.
--
-- Placeholders:
--
--     1: Params ID
--
-- Returns:
--
--     |scenario|perf|
--
SELECT runtime_stats.scenario AS scenario,
       (oracle_params.runtime / runtime_stats.mean) AS perf
FROM runtime_stats
LEFT JOIN oracle_params
ON runtime_stats.scenario=oracle_params.scenario
WHERE runtime_stats.params=?
