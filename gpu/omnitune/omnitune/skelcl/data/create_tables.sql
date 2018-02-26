/* METADATA */


-- Version table
CREATE TABLE version (
    version                         INTEGER,
    PRIMARY KEY (version)
);
INSERT INTO version VALUES (4);


/* KERNELS */


-- Kernels table
CREATE TABLE kernels (
    id                              CHAR(40),
    north                           INTEGER,
    south                           INTEGER,
    east                            INTEGER,
    west                            INTEGER,
    max_wg_size                     INTEGER,
    instruction_count               INTEGER,
    ratio_AShr_insts                REAL,
    ratio_Add_insts                 REAL,
    ratio_Alloca_insts              REAL,
    ratio_And_insts                 REAL,
    ratio_Br_insts                  REAL,
    ratio_Call_insts                REAL,
    ratio_FAdd_insts                REAL,
    ratio_FCmp_insts                REAL,
    ratio_FDiv_insts                REAL,
    ratio_FMul_insts                REAL,
    ratio_FPExt_insts               REAL,
    ratio_FPToSI_insts              REAL,
    ratio_FSub_insts                REAL,
    ratio_GetElementPtr_insts       REAL,
    ratio_ICmp_insts                REAL,
    ratio_InsertValue_insts         REAL,
    ratio_Load_insts                REAL,
    ratio_Mul_insts                 REAL,
    ratio_Or_insts                  REAL,
    ratio_PHI_insts                 REAL,
    ratio_Ret_insts                 REAL,
    ratio_SDiv_insts                REAL,
    ratio_SExt_insts                REAL,
    ratio_SIToFP_insts              REAL,
    ratio_SRem_insts                REAL,
    ratio_Select_insts              REAL,
    ratio_Shl_insts                 REAL,
    ratio_Store_insts               REAL,
    ratio_Sub_insts                 REAL,
    ratio_Trunc_insts               REAL,
    ratio_UDiv_insts                REAL,
    ratio_Xor_insts                 REAL,
    ratio_ZExt_insts                REAL,
    ratio_basic_blocks              REAL,
    ratio_memory_instructions       REAL,
    ratio_non_external_functions    REAL,
    PRIMARY KEY (id)
);

-- Kernel lookup table
CREATE TABLE kernel_lookup (
    north                           INTEGER,
    south                           INTEGER,
    east                            INTEGER,
    west                            INTEGER,
    max_wg_size                     INTEGER,
    source                          VARCHAR(32768),
    id                              CHAR(40)
);

-- Kernel names table
CREATE TABLE kernel_names (
    id                              CHAR(40),
    synthetic                       INTEGER,
    name                            VARCHAR(255),
    PRIMARY KEY (id)
);


/* DEVICES */


-- Devices table
CREATE TABLE devices (
    id                              VARCHAR(255),
    name                            VARCHAR(255),
    count                           INTEGER,
    address_bits                    INTEGER,
    double_fp_config                TEXT,
    endian_little                   TEXT,
    execution_capabilities          TEXT,
    extensions                      TEXT,
    global_mem_cache_size           INTEGER,
    global_mem_cache_type           TEXT,
    global_mem_cacheline_size       INTEGER,
    global_mem_size                 INTEGER,
    host_unified_memory             TEXT,
    image2d_max_height              INTEGER,
    image2d_max_width               INTEGER,
    image3d_max_depth               INTEGER,
    image3d_max_height              INTEGER,
    image3d_max_width               INTEGER,
    image_support                   TEXT,
    local_mem_size                  INTEGER,
    local_mem_type                  TEXT,
    max_clock_frequency             INTEGER,
    max_compute_units               INTEGER,
    max_constant_args               INTEGER,
    max_constant_buffer_size        INTEGER,
    max_mem_alloc_size              INTEGER,
    max_parameter_size              INTEGER,
    max_read_image_args             INTEGER,
    max_samplers                    INTEGER,
    max_work_group_size             INTEGER,
    max_work_item_dimensions        INTEGER,
    max_work_item_sizes_0           INTEGER,
    max_work_item_sizes_1           INTEGER,
    max_work_item_sizes_2           INTEGER,
    max_write_image_args            INTEGER,
    mem_base_addr_align             INTEGER,
    min_data_type_align_size        INTEGER,
    native_vector_width_char        INTEGER,
    native_vector_width_double      INTEGER,
    native_vector_width_float       INTEGER,
    native_vector_width_half        INTEGER,
    native_vector_width_int         INTEGER,
    native_vector_width_long        INTEGER,
    native_vector_width_short       INTEGER,
    preferred_vector_width_char     INTEGER,
    preferred_vector_width_double   INTEGER,
    preferred_vector_width_float    INTEGER,
    preferred_vector_width_half     INTEGER,
    preferred_vector_width_int      INTEGER,
    preferred_vector_width_long     INTEGER,
    preferred_vector_width_short    INTEGER,
    queue_properties                TEXT,
    single_fp_config                TEXT,
    type                            TEXT,
    vendor                          TEXT,
    vendor_id                       TEXT,
    version                         TEXT,
    PRIMARY KEY (id)
);

-- Devices lookup table
CREATE TABLE device_lookup (
    name                            VARCHAR(255),
    count                           INTEGER,
    id                              CHAR(40),
    PRIMARY KEY (name, count)
);


/* DATASETS */


-- Datasets table
CREATE TABLE datasets (
    id                              CHAR(40),
    width                           INTEGER,
    height                          INTEGER,
    tin                             VARCHAR(255),
    tout                            VARCHAR(255),
    PRIMARY KEY (id)
);

-- Datasets lookup table
CREATE TABLE dataset_lookup (
    width                           INTEGER,
    height                          INTEGER,
    tin                             VARCHAR(255),
    tout                            VARCHAR(255),
    id                              CHAR(40),
    PRIMARY KEY (width, height, tin, tout, id)
);


/* SCENARIOS */


-- Scenarios table
CREATE TABLE scenarios (
    id                              CHAR(40),
    device                          VARCHAR(255),
    kernel                          CHAR(40),
    dataset                         CHAR(40),
    PRIMARY KEY (id)
);


/* PARAMS */


-- Params table
CREATE TABLE params (
    id                              VARCHAR(255),
    wg_c                            INTEGER,
    wg_r                            INTEGER,
    PRIMARY KEY (id)
);


-- Refused params table
CREATE TABLE refused_params (
    scenario                        CHAR(40),
    params                          VARCHAR(255),
    PRIMARY KEY (scenario, params)
);

/* RUNTIMES */


-- Runtimes table
CREATE TABLE runtimes (
    scenario                        CHAR(40),
    params                          VARCHAR(255),
    runtime                         REAL
);


/* ORACLE TABLES */


-- Runtime stats table
CREATE TABLE runtime_stats (
    scenario                        CHAR(40),
    params                          VARCHAR(255),
    num_samples                     INTEGER,
    min                             REAL,
    mean                            REAL,
    max                             REAL,
    PRIMARY KEY (scenario, params)
);

-- Parameter stats table
CREATE TABLE param_stats (
    params                          VARCHAR(255), -- Key for params
    num_scenarios                   INTEGER,      -- Number of scenarios for which param is legal, 0 < num_scenarios
    coverage                        REAL,         -- num_scenarios / total number of scenarios, 0 < coverage <= 1
    performance                     REAL,         -- Geometric mean of performance relative to the oracle for all scenarios for which param was legal, 0 < performance <= 1
    PRIMARY KEY (params)
);

-- Scenario stats table
CREATE TABLE scenario_stats (
    scenario                        CHAR(40),     -- Key for scenarios
    num_params                      INTEGER,      -- The number of parameters in W_legal for scenario
    oracle_param                    VARCHAR(255), -- The best parameter
    oracle_runtime                  REAL,         -- The runtime of the best parameter
    worst_param                     VARCHAR(255), -- The worst parameter
    worst_runtime                   REAL,         -- The runtime of the worst parameter
    mean_runtime                    REAL,         -- The mean runtime of all parameters
    PRIMARY KEY (scenario)
);

-- Oracle parameters table
CREATE TABLE oracle_params (
    scenario                        CHAR(40),
    params                          VARCHAR(255),
    runtime                         REAL,
    PRIMARY KEY (scenario, params, runtime)
);

-- Variance stats table.
CREATE TABLE IF NOT EXISTS variance_stats (
    num_samples                     INTEGER,
    mean                            REAL,
    confinterval                    REAL,
    PRIMARY KEY (num_samples)
);


/* ML TABLES */


-- Classifiers table
CREATE TABLE classifiers (
    id                              VARCHAR(512),
    classname                       TEXT,
    options                         TEXT,
    PRIMARY KEY (id)
);


-- Error handlers table
CREATE TABLE err_fns (
    id                              VARCHAR(255),
    PRIMARY KEY (id)
);


-- Error handlers table
CREATE TABLE ml_datasets (
    id                              CHAR(40),
    data                            TEXT,     -- JSON dataset blob
    PRIMARY KEY (id)
);


-- ML evaluation jobs table
CREATE TABLE ml_jobs (
    id                              VARCHAR(255),     -- Descriptive job "name"
    PRIMARY KEY (id)
);


-- Classification results table
CREATE TABLE model_results (
    model                           TEXT,     -- Model name
    err_fn                          TEXT,     -- Key for err_fns.id
    scenario                        TEXT,     -- Key for scenarios
    actual                          TEXT,     -- Oracle params value, key for params
    predicted                       TEXT,     -- Predicted params value, key for params
    correct                         INTEGER,  -- 1 if prediction is correct, else 0
    illegal                         INTEGER,  -- 1 if prediction is >= max_wgsize, else 0
    refused                         INTEGER,  -- 1 if prediction is < max_wgsize and is refused, else 0
    performance                     REAL,     -- Performance relative to oracle, 0 <= performance <= 1
    speedup                         REAL,     -- Speedup over baseline, 0 <= speedup
    speedup_he                      REAL,     -- Speedup over human expert
    speedup_mo                      REAL      -- Speedup over mode param
);


-- Classification results table
CREATE TABLE classification_results (
    job                             TEXT,     -- Key for ml_jobs
    classifier                      TEXT,     -- Key for classifiers.id
    err_fn                          TEXT,     -- Key for err_fns.id
    dataset                         TEXT,     -- Key for datasets.id
    scenario                        TEXT,     -- Key for scenarios
    actual                          TEXT,     -- Oracle params value, key for params
    predicted                       TEXT,     -- Predicted params value, key for params
    baseline                        TEXT,     -- Baseline params value, key for params
    correct                         INTEGER,  -- 1 if prediction is correct, else 0
    illegal                         INTEGER,  -- 1 if prediction is >= max_wgsize, else 0
    refused                         INTEGER,  -- 1 if prediction is < max_wgsize and is refused, else 0
    performance                     REAL,     -- Performance relative to oracle, 0 <= performance <= 1
    speedup                         REAL,     -- Speedup over baseline, 0 <= speedup
    speedup_he                      REAL,     -- Speedup over human expert
    speedup_mo                      REAL,     -- Speedup over mode param
    time                            REAL
);


-- Table for results of classification using runtime regression.
CREATE TABLE runtime_classification_results (
    job                             TEXT,     -- Key for ml_jobs
    classifier                      TEXT,     -- Key for classifiers.id
    scenario                        TEXT,     -- Key for scenarios
    actual                          TEXT,     -- Oracle params value, key for params
    actual_runtime                  REAL,
    predicted                       TEXT,     -- Predicted params value, key for params
    predicted_runtime               REAL,
    actual_range                    REAL,
    predicted_range                 REAL,
    num_attempts                    INTEGER,  -- Number of attempts done
    correct                         INTEGER,  -- 1 if prediction is correct, else 0
    performance                     REAL,     -- Performance relative to oracle, 0 <= performance <= 1
    speedup                         REAL,     -- Speedup over baseline, 0 <= speedup
    speedup_he                      REAL,     -- Speedup over human expert
    speedup_mo                      REAL,     -- Speedup over mode param
    time                            REAL
);


-- Speedup regression results table
CREATE TABLE speedup_classification_results (
    job                             TEXT,     -- Key for ml_jobs
    classifier                      TEXT,     -- Key for classifiers.id
    scenario                        TEXT,     -- Key for scenarios
    actual                          TEXT,     -- Oracle params value, key for params
    actual_speedup                  REAL,
    predicted                       TEXT,     -- Predicted params value, key for params
    predicted_speedup               REAL,
    actual_range                    REAL,
    predicted_range                 REAL,
    num_attempts                    INTEGER,  -- Number of attempts done
    correct                         INTEGER,  -- 1 if prediction is correct, else 0
    performance                     REAL,     -- Performance relative to oracle, 0 <= performance <= 1
    speedup                         REAL,     -- Speedup over baseline, 0 <= speedup
    speedup_he                      REAL,     -- Speedup over human expert
    speedup_mo                      REAL,     -- Speedup over mode param
    time                            REAL
);
