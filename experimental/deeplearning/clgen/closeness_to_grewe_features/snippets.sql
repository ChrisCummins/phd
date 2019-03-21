# Overview table.

SELECT origin,
       count(len) AS n,
       round(avg(len)) AS avglen,
       max(len) AS maxlen
FROM
  (SELECT char_length(src) AS len,
          origin
   FROM static_features) t
GROUP BY origin
ORDER BY origin DESC;

# Overview of dynamic feature results.

SELECT origin,
       outcome,
       count(*) AS n,
       count(*) /
  (SELECT count(*)
   FROM dynamic_features) AS ratio
FROM dynamic_features AS df
LEFT JOIN static_features AS sf ON sf.id=df.static_features_id
GROUP BY origin,
         outcome;


SELECT origin,
       outcome,
       count(*) AS n,
       count(*) /
  (SELECT count(*)
   FROM
     (SELECT distinct(static_features_id)
      FROM dynamic_features) AS t) AS ratio
FROM
  (SELECT origin,
          outcome,
          static_features_id
   FROM dynamic_features AS df
   LEFT JOIN static_features ON df.static_features_id = static_features.id
   GROUP BY origin,
            outcome,
            static_features_id) AS t
GROUP BY origin,
         outcome; # Dynamic features summary: Grouped by device and outcome.


SELECT opencl_env,
       outcome,
       count(*) AS n
FROM
  (SELECT opencl_env,
          outcome,
          static_features_id
   FROM dynamic_features AS df
   GROUP BY opencl_env,
            outcome,
            static_features_id) AS t
GROUP BY opencl_env,
         outcome;

# Combined static and dynamic features values.

SELECT sf.origin,
       sf.src_sha256,
       sf.grewe_compute_operation_count AS compute_operation_count,
       sf.grewe_global_memory_access_count AS global_memory_access_count,
       sf.grewe_local_memory_access_count AS local_memory_access_count,
       sf.grewe_coalesced_memory_access_count AS coalesced_memory_access_count,
       df.gsize,
       df.wgsize,
       df.transferred_bytes,
       df.opencl_env,
       count(df.runtime_ms) AS runtime_count,
       avg(df.runtime_ms) AS avg_runtime,
       (max(df.runtime_ms) - min(df.runtime_ms)) / avg(df.runtime_ms) AS norm_range
FROM dynamic_features AS df
WHERE df.outcome = 'PASS'
  LEFT JOIN static_features AS sf ON sf.id=df.static_features_id
GROUP BY sf.origin,
         sf.src_sha256,
         sf.grewe_compute_operation_count,
         sf.grewe_global_memory_access_count,
         sf.grewe_local_memory_access_count,
         sf.grewe_coalesced_memory_access_count,
         df.wgsize,
         df.gsize,
         df.transferred_bytes,
         df.opencl_env;

# OpenCL sources containing kernels with unsupported arguments.

SELECT distinct(src)
FROM dynamic_features AS df
LEFT JOIN static_features AS sf ON df.static_features_id=sf.id
WHERE outcome = 'UNSUPPORTED_ARGUMENTS';

