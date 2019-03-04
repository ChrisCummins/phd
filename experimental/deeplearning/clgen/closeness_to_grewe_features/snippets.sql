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
       RESULT,
       count(*) AS n,
       count(*) /
  (SELECT count(*)
   FROM driver_result) AS ratio
FROM driver_result AS df
LEFT JOIN static_features AS sf ON sf.id=df.static_features_id
GROUP BY origin,
         RESULT;

# Combined static and raw features values.

SELECT sf.origin,
       sf.src_sha256,
       sf.src,
       sf.grewe_compute_operation_count AS compute_operation_count,
       sf.grewe_global_memory_access_count AS global_memory_access_count,
       sf.grewe_local_memory_access_count AS local_memory_access_count,
       sf.grewe_coalesced_memory_access_count AS coalesced_memory_access_count,
       df.wgsize,
       df.transferred_bytes,
       df.opencl_env,
       count(df.runtime_ms) AS runtime_count,
       avg(df.runtime_ms) AS avg_runtime,
       (max(df.runtime_ms) - min(df.runtime_ms)) / avg(df.runtime_ms) AS norm_range
FROM dynamic_features AS df
LEFT JOIN static_features AS sf ON sf.id=df.static_features_id
GROUP BY sf.src_sha256,
         sf.origin,
         sf.grewe_compute_operation_count,
         sf.grewe_global_memory_access_count,
         sf.grewe_local_memory_access_count,
         sf.grewe_coalesced_memory_access_count,
         df.wgsize,
         df.transferred_bytes,
         df.dataset,
         df.opencl_env;

# OpenCL sources containing kernels with unsupported arguments.

SELECT src
FROM driver_result AS dr
LEFT JOIN static_features AS sf ON dr.static_features_id=sf.id
WHERE RESULT = 'UNSUPPORTED_ARGUMENTS';

# Throwaway code when adding the '_<origin>' suffix to benchmark origins.

SELECT count(*)
FROM static_features
WHERE origin LIKE 'benchmarks%';


UPDATE static_features
SET origin='tmp_to_delete_benchmarks'
WHERE origin='benchmarks';


ALTER TABLE static_features MODIFY COLUMN origin varchar(128);


DELETE
FROM static_features
WHERE origin='tmp_to_delete_benchmarks';


UPDATE static_features
SET origin='clgen_vrae'
WHERE origin='clgen_legacy';


SELECT count(*)
FROM static_features
WHERE origin='clgen_legacy';


SELECT distinct(origin)
FROM static_features;