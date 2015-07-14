SELECT GEOMEAN(speedup) as mean_speedup
FROM classification_results
WHERE job="synthetic_real"
GROUP BY classifier,err_fn
ORDER BY mean_speedup DESC
LIMIT 1
