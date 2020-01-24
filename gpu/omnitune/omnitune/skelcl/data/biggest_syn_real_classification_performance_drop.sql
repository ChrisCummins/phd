-- Return the *largest* performance drop between classification jobs
-- using xval vs. synthetic_real. This gives a worse case indication
-- of the performance penalty from training using synthetic
-- benchmarks.
--

SELECT -- FIXME: Is geometric mean more appropriate?
 avg(xval.performance - synthetic_real.performance) * 100 AS performance_diff
FROM
  (SELECT *
   FROM classification_results
   WHERE job="synthetic_real" ) AS synthetic_real
LEFT JOIN
  (SELECT *
   FROM classification_results
   WHERE job="xval" ) AS xval ON synthetic_real.classifier = xval.classifier
AND synthetic_real.err_fn = xval.err_fn
AND synthetic_real.scenario = xval.scenario
GROUP BY synthetic_real.classifier,
         synthetic_real.err_fn
ORDER BY performance_diff DESC -- Rank by worst.
LIMIT 1;
