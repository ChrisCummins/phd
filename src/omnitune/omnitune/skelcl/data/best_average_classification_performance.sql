SELECT GEOMEAN(performance) * 100 as mean_performance
FROM classification_results
GROUP BY classifier,job
ORDER BY GEOMEAN(speedup) DESC
LIMIT 1
