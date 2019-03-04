-- Populate senario stats table.
--

DELETE
FROM scenario_stats;


INSERT INTO scenario_stats
SELECT scenarios.id AS scenario,

  (SELECT count(*)
   FROM runtime_stats
   WHERE scenario=scenarios.id ) AS num_params,

  (SELECT params
   FROM runtime_stats
   WHERE scenario=scenarios.id
     AND mean=
       (SELECT min(mean)
        FROM runtime_stats
        WHERE scenario=scenarios.id) ) AS oracle_param,

  (SELECT mean
   FROM runtime_stats
   WHERE scenario=scenarios.id
     AND mean=
       (SELECT min(mean)
        FROM runtime_stats
        WHERE scenario=scenarios.id) ) AS oracle_runtime,

  (SELECT params
   FROM runtime_stats
   WHERE scenario=scenarios.id
     AND mean=
       (SELECT max(mean)
        FROM runtime_stats
        WHERE scenario=scenarios.id) ) AS worst_param,

  (SELECT mean
   FROM runtime_stats
   WHERE scenario=scenarios.id
     AND mean=
       (SELECT max(mean)
        FROM runtime_stats
        WHERE scenario=scenarios.id) ) AS worst_runtime,

  (SELECT avg(mean)
   FROM runtime_stats
   WHERE scenario=scenarios.id ) AS mean_runtime
FROM scenarios;

