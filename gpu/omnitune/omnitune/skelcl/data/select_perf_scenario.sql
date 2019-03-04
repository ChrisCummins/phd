-- Return the performance of each param relative to the oracle for a
-- given scenario.
--
-- Placeholders:
--
--     1: Scenario ID
--     2: Scenario ID
--     3: Scenario ID
--
-- Returns:
--
--     |params|perf|
--

SELECT params,

  (SELECT mean AS oracle
   FROM runtime_stats
   WHERE scenario=?
     AND params=
       (SELECT params
        FROM oracle_params
        WHERE scenario=? ) ) / mean AS perf
FROM runtime_stats
WHERE scenario=?