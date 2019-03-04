-- Return the performance and speedup of the most successful
-- classification job, along with maximum speedup.
--

SELECT classifier,
       err_fn,
       avg(performance) * 100 AS mean_performance,
       geomean(speedup) AS mean_speedup,
       max(speedup) AS max_speedup,
       avg(speedup_he) AS mean_speedup_he,
       max(speedup_he) AS max_speedup_he,
       avg(speedup_mo) AS mean_speedup_mo,
       max(speedup_mo) AS max_speedup_mo
FROM classification_results
WHERE job="synthetic_real"
GROUP BY classifier,
         err_fn
ORDER BY mean_speedup DESC -- Rank by best speedup.
LIMIT 1