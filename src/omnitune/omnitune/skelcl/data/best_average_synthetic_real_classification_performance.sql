SELECT GEOMEAN(performance) * 100 as mean_performance
FROM classification_results
WHERE job="synthetic_real"
GROUP BY classifier,err_fn
ORDER BY GEOMEAN(speedup) DESC
LIMIT 1
