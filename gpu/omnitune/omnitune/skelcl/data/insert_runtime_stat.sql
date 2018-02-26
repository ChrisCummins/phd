-- Insert runtime statistics for a given scenario and params.
--
-- Placeholders:
--
--     1: Scenario ID
--     2: Params ID
--
INSERT INTO runtime_stats
SELECT
    scenario,
    params,
    COUNT(runtime),
    MIN(runtime),
    AVG(runtime),
    MAX(runtime)
FROM runtimes
WHERE scenario=?
AND params=?
