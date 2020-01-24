-- Merge contents of an attached database called "rhs".
 -------------
-- KERNELS --
-------------
 -- Kernels table

INSERT
OR
IGNORE INTO kernels
SELECT *
FROM rhs.kernels;

-- Kernel lookup table

INSERT
OR
IGNORE INTO kernel_lookup
SELECT *
FROM rhs.kernel_lookup;

-- Kernel names table

INSERT
OR
IGNORE INTO kernel_names
SELECT *
FROM rhs.kernel_names;

-------------
-- DEVICES --
-------------
 -- Devices table

INSERT
OR
IGNORE INTO devices
SELECT *
FROM rhs.devices;

-- Devices lookup table

INSERT
OR
IGNORE INTO device_lookup
SELECT *
FROM rhs.device_lookup;

--------------
-- DATASETS --
--------------
 -- Datasets table

INSERT
OR
IGNORE INTO datasets
SELECT *
FROM rhs.datasets;

-- Datasets lookup table

INSERT
OR
IGNORE INTO dataset_lookup
SELECT *
FROM rhs.dataset_lookup;

---------------
-- SCENARIOS --
---------------
 -- Scenarios table

INSERT
OR
IGNORE INTO scenarios
SELECT *
FROM rhs.scenarios;

------------
-- PARAMS --
------------
 -- Params table

INSERT
OR
IGNORE INTO params
SELECT *
FROM rhs.params;

--------------
-- RUNTIMES --
--------------
 -- Runtimes table

INSERT INTO runtimes
SELECT *
FROM rhs.runtimes;

-- Runtime stats table

DELETE
FROM rhs.runtime_stats;


INSERT INTO rhs.runtime_stats
SELECT scenario,
       params,
       count(runtime),
       min(runtime),
       avg(runtime),
       max(runtime)
FROM rhs.runtimes
GROUP BY scenario,
         params;
