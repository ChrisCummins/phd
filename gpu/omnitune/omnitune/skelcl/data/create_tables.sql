/* METADATA */ -- Version table

CREATE TABLE VERSION (VERSION INTEGER, PRIMARY KEY (VERSION));


INSERT INTO VERSION
VALUES (4);

/* KERNELS */ -- Kernels table

CREATE TABLE kernels (id char(40),
                         north INTEGER, south INTEGER, east INTEGER, west INTEGER, max_wg_size INTEGER, instruction_count INTEGER, ratio_ashr_insts REAL, ratio_add_insts REAL, ratio_alloca_insts REAL, ratio_and_insts REAL, ratio_br_insts REAL, ratio_call_insts REAL, ratio_fadd_insts REAL, ratio_fcmp_insts REAL, ratio_fdiv_insts REAL, ratio_fmul_insts REAL, ratio_fpext_insts REAL, ratio_fptosi_insts REAL, ratio_fsub_insts REAL, ratio_getelementptr_insts REAL, ratio_icmp_insts REAL, ratio_insertvalue_insts REAL, ratio_load_insts REAL, ratio_mul_insts REAL, ratio_or_insts REAL, ratio_phi_insts REAL, ratio_ret_insts REAL, ratio_sdiv_insts REAL, ratio_sext_insts REAL, ratio_sitofp_insts REAL, ratio_srem_insts REAL, ratio_select_insts REAL, ratio_shl_insts REAL, ratio_store_insts REAL, ratio_sub_insts REAL, ratio_trunc_insts REAL, ratio_udiv_insts REAL, ratio_xor_insts REAL, ratio_zext_insts REAL, ratio_basic_blocks REAL, ratio_memory_instructions REAL, ratio_non_external_functions REAL, PRIMARY KEY (id));

-- Kernel lookup table

CREATE TABLE kernel_lookup (north INTEGER, south INTEGER, east INTEGER, west INTEGER, max_wg_size INTEGER, SOURCE varchar(32768),
                                                                                                                  id char(40));

-- Kernel names table

CREATE TABLE kernel_names (id char(40),
                              synthetic INTEGER, name varchar(255),
                                                      PRIMARY KEY (id));

/* DEVICES */ -- Devices table

CREATE TABLE devices (id varchar(255),
                         name varchar(255),
                              COUNT INTEGER, address_bits INTEGER, double_fp_config TEXT, endian_little TEXT, execution_capabilities TEXT, extensions TEXT, global_mem_cache_size INTEGER, global_mem_cache_type TEXT, global_mem_cacheline_size INTEGER, global_mem_size INTEGER, host_unified_memory TEXT, image2d_max_height INTEGER, image2d_max_width INTEGER, image3d_max_depth INTEGER, image3d_max_height INTEGER, image3d_max_width INTEGER, image_support TEXT, local_mem_size INTEGER, local_mem_type TEXT, max_clock_frequency INTEGER, max_compute_units INTEGER, max_constant_args INTEGER, max_constant_buffer_size INTEGER, max_mem_alloc_size INTEGER, max_parameter_size INTEGER, max_read_image_args INTEGER, max_samplers INTEGER, max_work_group_size INTEGER, max_work_item_dimensions INTEGER, max_work_item_sizes_0 INTEGER, max_work_item_sizes_1 INTEGER, max_work_item_sizes_2 INTEGER, max_write_image_args INTEGER, mem_base_addr_align INTEGER, min_data_type_align_size INTEGER, native_vector_width_char INTEGER, native_vector_width_double INTEGER, native_vector_width_float INTEGER, native_vector_width_half INTEGER, native_vector_width_int INTEGER, native_vector_width_long INTEGER, native_vector_width_short INTEGER, preferred_vector_width_char INTEGER, preferred_vector_width_double INTEGER, preferred_vector_width_float INTEGER, preferred_vector_width_half INTEGER, preferred_vector_width_int INTEGER, preferred_vector_width_long INTEGER, preferred_vector_width_short INTEGER, queue_properties TEXT, single_fp_config TEXT, TYPE TEXT, vendor TEXT, vendor_id TEXT, VERSION TEXT, PRIMARY KEY (id));

-- Devices lookup table

CREATE TABLE device_lookup (name varchar(255),
                                 COUNT INTEGER, id char(40),
                                                   PRIMARY KEY (name,
                                                                COUNT));

/* DATASETS */ -- Datasets table

CREATE TABLE datasets (id char(40),
                          width INTEGER, height INTEGER, tin varchar(255),
                                                             tout varchar(255),
                                                                  PRIMARY KEY (id));

-- Datasets lookup table

CREATE TABLE dataset_lookup (width INTEGER, height INTEGER, tin varchar(255),
                                                                tout varchar(255),
                                                                     id char(40),
                                                                        PRIMARY KEY (width,
                                                                                     height,
                                                                                     tin,
                                                                                     tout,
                                                                                     id));

/* SCENARIOS */ -- Scenarios table

CREATE TABLE scenarios (id char(40),
                           device varchar(255),
                                  kernel char(40),
                                         dataset char(40),
                                                 PRIMARY KEY (id));

/* PARAMS */ -- Params table

CREATE TABLE params (id varchar(255),
                        wg_c INTEGER, wg_r INTEGER, PRIMARY KEY (id));

-- Refused params table

CREATE TABLE refused_params (scenario char(40),
                                      params varchar(255),
                                             PRIMARY KEY (scenario,
                                                          params));

/* RUNTIMES */ -- Runtimes table

CREATE TABLE runtimes (scenario char(40),
                                params varchar(255),
                                       runtime REAL);

/* ORACLE TABLES */ -- Runtime stats table

CREATE TABLE runtime_stats (scenario char(40),
                                     params varchar(255),
                                            num_samples INTEGER, MIN REAL, mean REAL, MAX REAL, PRIMARY KEY (scenario,
                                                                                                             params));

-- Parameter stats table

CREATE TABLE param_stats (params varchar(255), -- Key for params
 num_scenarios INTEGER, -- Number of scenarios for which param is legal, 0 < num_scenarios
 coverage REAL, -- num_scenarios / total number of scenarios, 0 < coverage <= 1
 performance REAL, -- Geometric mean of performance relative to the oracle for all scenarios for which param was legal, 0 < performance <= 1
 PRIMARY KEY (params));

-- Scenario stats table

CREATE TABLE scenario_stats (scenario char(40), -- Key for scenarios
 num_params INTEGER, -- The number of parameters in W_legal for scenario
 oracle_param varchar(255), -- The best parameter
 oracle_runtime REAL, -- The runtime of the best parameter
 worst_param varchar(255), -- The worst parameter
 worst_runtime REAL, -- The runtime of the worst parameter
 mean_runtime REAL, -- The mean runtime of all parameters
 PRIMARY KEY (scenario));

-- Oracle parameters table

CREATE TABLE oracle_params (scenario char(40),
                                     params varchar(255),
                                            runtime REAL, PRIMARY KEY (scenario,
                                                                       params,
                                                                       runtime));

-- Variance stats table.

CREATE TABLE IF NOT EXISTS variance_stats (num_samples INTEGER, mean REAL, confinterval REAL, PRIMARY KEY (num_samples));

/* ML TABLES */ -- Classifiers table

CREATE TABLE classifiers (id varchar(512),
                             classname TEXT, OPTIONS TEXT, PRIMARY KEY (id));

-- Error handlers table

CREATE TABLE err_fns (id varchar(255),
                         PRIMARY KEY (id));

-- Error handlers table

CREATE TABLE ml_datasets (id char(40),
                             DATA TEXT, -- JSON dataset blob
 PRIMARY KEY (id));

-- ML evaluation jobs table

CREATE TABLE ml_jobs (id varchar(255), -- Descriptive job "name"
 PRIMARY KEY (id));

-- Classification results table

CREATE TABLE model_results (model TEXT, -- Model name
 err_fn TEXT, -- Key for err_fns.id
 scenario TEXT, -- Key for scenarios
 actual TEXT, -- Oracle params value, key for params
 predicted TEXT, -- Predicted params value, key for params
 correct INTEGER, -- 1 if prediction is correct, else 0
 illegal INTEGER, -- 1 if prediction is >= max_wgsize, else 0
 refused INTEGER, -- 1 if prediction is < max_wgsize and is refused, else 0
 performance REAL, -- Performance relative to oracle, 0 <= performance <= 1
 speedup REAL, -- Speedup over baseline, 0 <= speedup
 speedup_he REAL, -- Speedup over human expert
 speedup_mo REAL -- Speedup over mode param
);

-- Classification results table

CREATE TABLE classification_results (job TEXT, -- Key for ml_jobs
 classifier TEXT, -- Key for classifiers.id
 err_fn TEXT, -- Key for err_fns.id
 dataset TEXT, -- Key for datasets.id
 scenario TEXT, -- Key for scenarios
 actual TEXT, -- Oracle params value, key for params
 predicted TEXT, -- Predicted params value, key for params
 baseline TEXT, -- Baseline params value, key for params
 correct INTEGER, -- 1 if prediction is correct, else 0
 illegal INTEGER, -- 1 if prediction is >= max_wgsize, else 0
 refused INTEGER, -- 1 if prediction is < max_wgsize and is refused, else 0
 performance REAL, -- Performance relative to oracle, 0 <= performance <= 1
 speedup REAL, -- Speedup over baseline, 0 <= speedup
 speedup_he REAL, -- Speedup over human expert
 speedup_mo REAL, -- Speedup over mode param
 TIME REAL);

-- Table for results of classification using runtime regression.

CREATE TABLE runtime_classification_results (job TEXT, -- Key for ml_jobs
 classifier TEXT, -- Key for classifiers.id
 scenario TEXT, -- Key for scenarios
 actual TEXT, -- Oracle params value, key for params
 actual_runtime REAL, predicted TEXT, -- Predicted params value, key for params
 predicted_runtime REAL, actual_range REAL, predicted_range REAL, num_attempts INTEGER, -- Number of attempts done
 correct INTEGER, -- 1 if prediction is correct, else 0
 performance REAL, -- Performance relative to oracle, 0 <= performance <= 1
 speedup REAL, -- Speedup over baseline, 0 <= speedup
 speedup_he REAL, -- Speedup over human expert
 speedup_mo REAL, -- Speedup over mode param
 TIME REAL);

-- Speedup regression results table

CREATE TABLE speedup_classification_results (job TEXT, -- Key for ml_jobs
 classifier TEXT, -- Key for classifiers.id
 scenario TEXT, -- Key for scenarios
 actual TEXT, -- Oracle params value, key for params
 actual_speedup REAL, predicted TEXT, -- Predicted params value, key for params
 predicted_speedup REAL, actual_range REAL, predicted_range REAL, num_attempts INTEGER, -- Number of attempts done
 correct INTEGER, -- 1 if prediction is correct, else 0
 performance REAL, -- Performance relative to oracle, 0 <= performance <= 1
 speedup REAL, -- Speedup over baseline, 0 <= speedup
 speedup_he REAL, -- Speedup over human expert
 speedup_mo REAL, -- Speedup over mode param
 TIME REAL);

