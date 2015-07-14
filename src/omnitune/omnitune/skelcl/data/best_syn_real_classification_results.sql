-- Return the performance and speedup of the most successful
-- classification job, along with maximum speedup.
--
SELECT
    classifier,
    err_fn,
    AVG(performance) * 100 AS mean_performance,
    GEOMEAN(speedup) AS mean_speedup,
    MAX(speedup) AS max_speedup
FROM classification_results
WHERE job="synthetic_real"
GROUP BY classifier,err_fn
ORDER BY mean_speedup DESC -- Rank by best speedup.
LIMIT 1
