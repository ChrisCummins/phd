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
