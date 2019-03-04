SELECT performance
FROM
  (SELECT (scenario_stats.oracle_runtime / runtime_stats.mean) AS performance,
          (params.wg_r * 1.0 / kernels.max_wg_size) * 100 AS ratio_max_wgsize
   FROM runtime_stats
   LEFT JOIN scenarios ON runtime_stats.scenario=scenarios.id
   LEFT JOIN kernels ON scenarios.kernel=kernels.id
   LEFT JOIN params ON runtime_stats.params=params.id
   LEFT JOIN scenario_stats ON runtime_stats.scenario=scenario_stats.scenario)
WHERE ratio_max_wgsize > ?
  AND ratio_max_wgsize <= ?