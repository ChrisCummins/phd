.mode csv .header on .out "version.csv"
SELECT *
FROM VERSION;

.out "scenarios.csv"
SELECT *
FROM scenarios;

.out "kernels.csv"
SELECT *
FROM kernels;

.out "kernels.csv"
SELECT *
FROM kernels;

.out "kernel_lookup.csv"
SELECT *
FROM kernel_lookup;

.out "devices.csv"
SELECT *
FROM devices;

.out "device_lookup.csv"
SELECT *
FROM device_lookup;

.out "datasets.csv"
SELECT *
FROM datasets;

.out "dataset_lookup.csv"
SELECT *
FROM dataset_lookup;

.out "params.csv"
SELECT *
FROM params;

.out "runtimes.csv"
SELECT *
FROM runtimes;

.out "runtime_stats.csv"
SELECT *
FROM runtime_stats;

.out "oracle_params.csv"
SELECT *
FROM oracle_params;

.out "classifiers.csv"
SELECT *
FROM classifiers;

.out "err_fns.csv"
SELECT *
FROM err_fns;

.out "ml_datasets.csv"
SELECT *
FROM ml_datasets;

.out "ml_jobs.csv"
SELECT *
FROM ml_jobs;

.out "classification_results.csv"
SELECT *
FROM classification_results;

.out "runtime_regression_results.csv"
SELECT *
FROM runtime_regression_results;

.out "runtime_classification_results.csv"
SELECT *
FROM runtime_classification_results;

.out "speedup_regression_results.csv"
SELECT *
FROM speedup_regression_results;

.out "speedup_classification_results.csv"
SELECT *
FROM speedup_classification_results;

