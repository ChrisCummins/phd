SELECT GEOMEAN(performance) * 100 as mean_performance
FROM classification_results
GROUP BY job,classifier,err_fn
ORDER BY GEOMEAN(speedup) DESC
LIMIT 1
