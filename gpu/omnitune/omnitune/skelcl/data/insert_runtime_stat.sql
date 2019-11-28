-- Insert runtime statistics for a given scenario and params.
--
-- Placeholders:
--
--     1: Scenario ID
--     2: Params ID
--

INSERT INTO runtime_stats
SELECT scenario,
       params,
       count(runtime),
       min(runtime),
       avg(runtime),
       max(runtime)
FROM runtimes
WHERE scenario=?
  AND params=?;

