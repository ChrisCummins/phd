-- Populate senario stats table.
--
DELETE FROM scenario_stats;

INSERT INTO scenario_stats
SELECT
    scenarios.id AS scenario,
    (
        SELECT Count(*)
        FROM runtime_stats
        WHERE scenario=scenarios.id
    ) AS num_params,
    (
        SELECT params
        FROM runtime_stats
        WHERE
            scenario=scenarios.id AND
            mean=(SELECT MIN(mean) FROM runtime_stats WHERE scenario=scenarios.id)
    ) AS oracle_param,
    (
        SELECT mean
        FROM runtime_stats
        WHERE
            scenario=scenarios.id AND
            mean=(SELECT MIN(mean) FROM runtime_stats WHERE scenario=scenarios.id)
    ) AS oracle_runtime,
    (
        SELECT params
        FROM runtime_stats
        WHERE
            scenario=scenarios.id AND
            mean=(SELECT MAX(mean) FROM runtime_stats WHERE scenario=scenarios.id)
    ) AS worst_param,
    (
        SELECT mean
        FROM runtime_stats
        WHERE
            scenario=scenarios.id AND
            mean=(SELECT MAX(mean) FROM runtime_stats WHERE scenario=scenarios.id)
    ) AS worst_runtime,
    (
        SELECT AVG(mean)
        FROM runtime_stats
        WHERE scenario=scenarios.id
    ) AS mean_runtime
FROM scenarios;
