--------------
-- METADATA --
--------------

-- Version table
CREATE TABLE version (
    version                         INTEGER,
    PRIMARY KEY (version)
);
INSERT INTO version VALUES (4);


-------------
-- KERNELS --
-------------

-- Kernels table
CREATE TABLE kernels (
    id                              TEXT,
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
    source                          TEXT,
    id                              TEXT,
    PRIMARY KEY (north, south, east, west, max_wg_size, source)
);

-- Kernel names table
CREATE TABLE kernel_names (
    id                              TEXT,
    synthetic                       INTEGER,
    name                            TEXT,
    PRIMARY KEY (id)
);


-------------
-- DEVICES --
-------------

-- Devices table
CREATE TABLE devices (
    id                              TEXT,
    name                            TEXT,
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
    name                            TEXT,
    count                           INTEGER,
    id                              TEXT,
    PRIMARY KEY (name, count)
);


--------------
-- DATASETS --
--------------

-- Datasets table
CREATE TABLE datasets (
    id                              TEXT,
    width                           INTEGER,
    height                          INTEGER,
    tin                             TEXT,
    tout                            TEXT,
    PRIMARY KEY (id)
);

-- Datasets lookup table
CREATE TABLE dataset_lookup (
    width                           INTEGER,
    height                          INTEGER,
    tin                             TEXT,
    tout                            TEXT,
    id                              TEXT,
    PRIMARY KEY (width, height, tin, tout, id)
);


---------------
-- SCENARIOS --
---------------

-- Scenarios table
CREATE TABLE scenarios (
    id                              TEXT,
    device                          TEXT,
    kernel                          TEXT,
    dataset                         TEXT,
    PRIMARY KEY (id)
);


------------
-- PARAMS --
------------

-- Params table
CREATE TABLE params (
    id                              TEXT,
    wg_c                            INTEGER,
    wg_r                            INTEGER,
    PRIMARY KEY (id)
);


--------------
-- RUNTIMES --
--------------

-- Runtimes table
CREATE TABLE runtimes (
    scenario                        TEXT,
    params                          TEXT,
    runtime                         REAL
);


-------------------
-- ORACLE TABLES --
-------------------

-- Runtime stats table
CREATE TABLE runtime_stats (
    scenario                        TEXT,
    params                          TEXT,
    num_samples                     INTEGER,
    min                             REAL,
    mean                            REAL,
    max                             REAL,
    PRIMARY KEY (scenario, params)
);

-- Oracle parameters table
CREATE TABLE oracle_params (
    scenario                        TEXT,
    params                          TEXT,
    runtime                         REAL,
    PRIMARY KEY (scenario, params, runtime)
);


---------------
-- ML TABLES --
---------------

-- Classifiers table
CREATE TABLE classifiers (
    id                              TEXT,
    classname                       TEXT,
    options                         TEXT,
    PRIMARY KEY (id)
);


-- Error handlers table
CREATE TABLE err_fns (
    id                              TEXT,
    PRIMARY KEY (id)
);


-- Error handlers table
CREATE TABLE ml_datasets (
    id                              TEXT,
    data                            TEXT,     -- JSON dataset blob
    PRIMARY KEY (id)
);


-- ML evaluation jobs table
CREATE TABLE ml_jobs (
    id                              TEXT,     -- Descriptive job "name"
    PRIMARY KEY (id)
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
    invalid                         INTEGER,  -- 1 if *first* prediction was valid, else 0
    performance                     REAL,     -- Performance relative to oracle, 0 <= performance <= 1
    speedup                         REAL      -- Speedup over baseline, 0 <= speedup
);


-- Regression results table
CREATE TABLE runtime_regression_results (
    job                             TEXT,     -- Key for ml_jobs
    classifier                      TEXT,     -- Key for classifiers.id
    dataset                         TEXT,     -- Key for datasets.id
    scenario                        TEXT,     -- Key for scenarios
    actual                          REAL,     -- Actual runtime value
    predicted                       REAL,     -- Predicted runtime value
    norm_predicted                  REAL,     -- Predicted runtime, normalise to actual runtime
    norm_err                        REAL      -- abs(norm_predicted - 1)
);
