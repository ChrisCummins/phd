-- Return the performance of each *legal* scenario relative to the
-- oracle for a given param.
--
-- Placeholders:
--
--     1: Params ID
--
-- Returns:
--
--     |scenario|perf|
--

SELECT stats.scenario AS scenario,
       (oracle.runtime / stats.mean) AS perf
FROM runtime_stats AS stats
LEFT JOIN oracle_params AS oracle ON stats.scenario=oracle.scenario
WHERE stats.params={} WHERE stats.scenario IN ({})
