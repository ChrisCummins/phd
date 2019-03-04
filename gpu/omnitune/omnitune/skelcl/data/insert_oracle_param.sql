-- Insert oracle params for a given scenario.
--
-- Placeholders:
--
--     1: Scenario ID
--     2: Scenario ID
--
-- Returns:
--
--     |scenario|params|runtime|
--

INSERT INTO oracle_params
SELECT scenario,
       params,
       mean AS runtime
FROM runtime_stats
WHERE scenario=?
  AND mean=
    (SELECT min(mean)
     FROM runtime_stats
     WHERE scenario=?)