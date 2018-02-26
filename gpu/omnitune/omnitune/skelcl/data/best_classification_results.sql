-- Return the performance and speedup of the most successful
-- classification job, along with maximum speedup.
--
SELECT
    classifier,
    err_fn,
    AVG(performance) * 100 AS mean_performance,
    GEOMEAN(speedup) AS mean_speedup,
    MAX(speedup) AS max_speedup,
    AVG(speedup_he) AS mean_speedup_he,
    MAX(speedup_he) AS max_speedup_he,
    AVG(speedup_mo) AS mean_speedup_mo,
    MAX(speedup_mo) AS max_speedup_mo
FROM classification_results
GROUP BY job,classifier,err_fn
ORDER BY mean_speedup DESC -- Rank by best speedup.
LIMIT 1
