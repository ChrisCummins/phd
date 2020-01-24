-- Return the performance of each param relative to the oracle for a
-- given scenario.
--
-- Placeholders:
--
--     1: Parameters ID
--     2: Scenario ID
--
-- Returns:
--
--     |ratio_max_wgsize|
--

SELECT
  (SELECT wg_c * wg_r
   FROM params
   WHERE id=? ) / cast(
                         (SELECT kernels.max_wg_size
                          FROM scenarios
                          LEFT JOIN kernels ON scenarios.kernel=kernels.id
                          WHERE scenarios.id=? ) AS real) AS ratio_max_wgsize;
