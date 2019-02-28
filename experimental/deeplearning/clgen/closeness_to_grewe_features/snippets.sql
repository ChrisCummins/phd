# Overview table.
SELECT
	origin, 
    COUNT(len) AS n, 
    ROUND(AVG(len)) AS avglen,
    MAX(len) AS maxlen
FROM  (
	SELECT 
		CHAR_LENGTH(src) as len, 
        origin 
	FROM 
		static_features
) t 
GROUP BY origin 
ORDER BY origin DESC;

# Overview of dynamic feature results.
SELECT
	origin,
	result,
    COUNT(*) as n,
    COUNT(*) / (SELECT COUNT(*) FROM driver_result) as ratio
FROM
	driver_result AS df
LEFT JOIN
	static_features AS sf
ON sf.id=df.static_features_id
GROUP BY
	origin,
	result;

# Combined static and raw features values.
SELECT
    sf.origin,
    sf.src_sha256,
    sf.src,
    sf.grewe_compute_operation_count as compute_operation_count,
    sf.grewe_global_memory_access_count as global_memory_access_count,
    sf.grewe_local_memory_access_count as local_memory_access_count,
    sf.grewe_coalesced_memory_access_count as coalesced_memory_access_count,
    df.wgsize,
    df.transferred_bytes, 
	df.opencl_env,
    COUNT(df.runtime_ms) as runtime_count,
    AVG(df.runtime_ms) as avg_runtime,
    (MAX(df.runtime_ms) - MIN(df.runtime_ms)) / AVG(df.runtime_ms) as norm_range
FROM
	dynamic_features AS df
LEFT JOIN
	static_features AS sf
ON sf.id=df.static_features_id
GROUP BY
	sf.src_sha256,
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
SELECT
	src
FROM
	driver_result as dr
LEFT JOIN
	static_features as sf
ON dr.static_features_id=sf.id
WHERE
	result = 'UNSUPPORTED_ARGUMENTS';