-- Populate runtime stats table from runtime.
--
DELETE FROM runtime_stats;

INSERT INTO runtime_stats
SELECT
    scenario,
    params,
    Count(runtime),
    MIN(runtime),
    AVG(runtime),
    MAX(runtime)
FROM runtimes
GROUP BY scenario,params;
