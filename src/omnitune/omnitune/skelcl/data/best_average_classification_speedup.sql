SELECT GEOMEAN(speedup) as mean_speedup
FROM classification_results
GROUP BY classifier,job
ORDER BY mean_speedup DESC
LIMIT 1
