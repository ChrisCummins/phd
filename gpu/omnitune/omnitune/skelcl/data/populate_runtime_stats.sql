-- Populate runtime stats table from runtime.
--

DELETE
FROM runtime_stats;


INSERT INTO runtime_stats
SELECT scenario,
       params,
       count(runtime),
       min(runtime),
       avg(runtime),
       max(runtime)
FROM runtimes
GROUP BY scenario,
         params;

