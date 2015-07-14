SELECT GEOMEAN(speedup) as mean_speedup
FROM classification_results
GROUP BY job,classifier,err_fn
ORDER BY mean_speedup DESC
LIMIT 1
