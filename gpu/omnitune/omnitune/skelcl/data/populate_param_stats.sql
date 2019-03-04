-- Populate param stats table.
--

DELETE
FROM param_stats;


INSERT INTO param_stats
SELECT id AS params,

  (SELECT count(*) AS REAL
   FROM runtime_stats
   WHERE params=id ) AS num_scenarios,

  (SELECT cast(count(*) AS REAL) /
     (SELECT count(*)
      FROM scenarios)
   FROM runtime_stats
   WHERE params=id ) AS coverage,

  (SELECT avg(oracle.runtime / stats.mean)
   FROM runtime_stats AS stats
   LEFT JOIN oracle_params AS oracle ON stats.scenario=oracle.scenario
   WHERE stats.params=id ) AS performance
FROM params;

