--
-- Fetched
--

CREATE TABLE IF NOT EXISTS Repositories (
       url                 TEXT NOT NULL,       -- GitHub URL
       owner               TEXT,                -- Email
       name                TEXT NOT NULL,       -- GitHub Repository name
       fork                INT NOT NULL,        -- 1 if repo is fork, else 0
       stars               INT NOT NULL,        -- GitHub stargazer count
       contributors        INT NOT NULL,        -- Number of contributors
       forks               INT NOT NULL,        -- Number of forks
       created_at          DATETIME NOT NULL,   -- Timestamp created
       updated_at          DATETIME NOT NULL,   -- Timestamp last updated
       UNIQUE(url)
);

CREATE TABLE IF NOT EXISTS ContentFiles (
       url                 TEXT NOT NULL,       -- GitHub URL
       path                TEXT NOT NULL,       -- File path within repo
       repo_url            TEXT NOT NULL,       -- Repositories.url
       contents            TEXT NOT NULL,       -- File contents
       sha                 TEXT NOT NULL,       -- File checksum
       size                INT NOT NULL,        -- File size
       UNIQUE(url)
);

--
-- Preprocessed
--

CREATE TABLE IF NOT EXISTS Preprocessed (
       sha                 TEXT NOT NULL,       -- OpenCLFile.sha
       contents            TEXT NOT NULL,       -- Preprocessed
       UNIQUE(sha)
);

CREATE TABLE IF NOT EXISTS PreprocessedErrors (
       sha                 TEXT NOT NULL,       -- OpenCLFile.sha
       message             TEXT NOT NULL,       -- Compilation error
       UNIQUE(sha)
);

CREATE TABLE IF NOT EXISTS Bytecodes (
       sha                 TEXT NOT NULL,       -- OpenCLFile.sha
       contents            TEXT NOT NULL,       -- Bytecode
       UNIQUE(sha)
);

CREATE TABLE IF NOT EXISTS BytecodeErrors (
       sha                 TEXT NOT NULL,       -- OpenCLFile.sha
       message             TEXT NOT NULL,       -- Compilation error
       UNIQUE(sha)
);


--
-- Bytecode Features
--

CREATE TABLE IF NOT EXISTS BytecodeFeatures (
        sha                        TEXT NOT NULL, -- OpenCLFile.sha
        instructions_of_all_types  INTEGER DEFAULT 0,
        basic_blocks               INTEGER DEFAULT 0,
        memory_instructions        INTEGER DEFAULT 0,
        non_external_functions     INTEGER DEFAULT 0,
        -- Specific instruction counts:
        Add_insts                  INTEGER DEFAULT 0,
        Alloca_insts               INTEGER DEFAULT 0,
        And_insts                  INTEGER DEFAULT 0,
        AShr_insts                 INTEGER DEFAULT 0,
        BitCast_insts              INTEGER DEFAULT 0,
        Br_insts                   INTEGER DEFAULT 0,
        Call_insts                 INTEGER DEFAULT 0,
        ExtractElement_insts       INTEGER DEFAULT 0,
        ExtractValue_insts         INTEGER DEFAULT 0,
        FAdd_insts                 INTEGER DEFAULT 0,
        FCmp_insts                 INTEGER DEFAULT 0,
        FDiv_insts                 INTEGER DEFAULT 0,
        FMul_insts                 INTEGER DEFAULT 0,
        FPExt_insts                INTEGER DEFAULT 0,
        FPToSI_insts               INTEGER DEFAULT 0,
        FPToUI_insts               INTEGER DEFAULT 0,
        FPTrunc_insts              INTEGER DEFAULT 0,
        FSub_insts                 INTEGER DEFAULT 0,
        GetElementPtr_insts        INTEGER DEFAULT 0,
        ICmp_insts                 INTEGER DEFAULT 0,
        InsertElement_insts        INTEGER DEFAULT 0,
        InsertValue_insts          INTEGER DEFAULT 0,
        IntToPtr_insts             INTEGER DEFAULT 0,
        Load_insts                 INTEGER DEFAULT 0,
        LShr_insts                 INTEGER DEFAULT 0,
        Mul_insts                  INTEGER DEFAULT 0,
        Or_insts                   INTEGER DEFAULT 0,
        PHI_insts                  INTEGER DEFAULT 0,
        PtrToInt_insts             INTEGER DEFAULT 0,
        Ret_insts                  INTEGER DEFAULT 0,
        SDiv_insts                 INTEGER DEFAULT 0,
        Select_insts               INTEGER DEFAULT 0,
        SExt_insts                 INTEGER DEFAULT 0,
        Shl_insts                  INTEGER DEFAULT 0,
        ShuffleVector_insts        INTEGER DEFAULT 0,
        SIToFP_insts               INTEGER DEFAULT 0,
        SRem_insts                 INTEGER DEFAULT 0,
        Store_insts                INTEGER DEFAULT 0,
        Sub_insts                  INTEGER DEFAULT 0,
        Switch_insts               INTEGER DEFAULT 0,
        Trunc_insts                INTEGER DEFAULT 0,
        UDiv_insts                 INTEGER DEFAULT 0,
        UIToFP_insts               INTEGER DEFAULT 0,
        URem_insts                 INTEGER DEFAULT 0,
        Xor_insts                  INTEGER DEFAULT 0,
        ZExt_insts                 INTEGER DEFAULT 0,
        -- Ratios:
        ratio_basic_blocks         REAL DEFAULT 0.0,
        ratio_memory_instructions  REAL DEFAULT 0.0,
        ratio_non_external_functions REAL DEFAULT 0.0,
        -- Specific instruction ratios:
        ratio_Add_insts            REAL DEFAULT 0.0,
        ratio_Alloca_insts         REAL DEFAULT 0.0,
        ratio_And_insts            REAL DEFAULT 0.0,
        ratio_AShr_insts           REAL DEFAULT 0.0,
        ratio_BitCast_insts        REAL DEFAULT 0.0,
        ratio_Br_insts             REAL DEFAULT 0.0,
        ratio_Call_insts           REAL DEFAULT 0.0,
        ratio_ExtractElement_insts REAL DEFAULT 0.0,
        ratio_ExtractValue_insts   REAL DEFAULT 0.0,
        ratio_FAdd_insts           REAL DEFAULT 0.0,
        ratio_FCmp_insts           REAL DEFAULT 0.0,
        ratio_FDiv_insts           REAL DEFAULT 0.0,
        ratio_FMul_insts           REAL DEFAULT 0.0,
        ratio_FPExt_insts          REAL DEFAULT 0.0,
        ratio_FPToSI_insts         REAL DEFAULT 0.0,
        ratio_FPToUI_insts         REAL DEFAULT 0.0,
        ratio_FPTrunc_insts        REAL DEFAULT 0.0,
        ratio_FSub_insts           REAL DEFAULT 0.0,
        ratio_GetElementPtr_insts  REAL DEFAULT 0.0,
        ratio_ICmp_insts           REAL DEFAULT 0.0,
        ratio_InsertElement_insts  REAL DEFAULT 0.0,
        ratio_InsertValue_insts    REAL DEFAULT 0.0,
        ratio_IntToPtr_insts       REAL DEFAULT 0.0,
        ratio_Load_insts           REAL DEFAULT 0.0,
        ratio_LShr_insts           REAL DEFAULT 0.0,
        ratio_Mul_insts            REAL DEFAULT 0.0,
        ratio_Or_insts             REAL DEFAULT 0.0,
        ratio_PHI_insts            REAL DEFAULT 0.0,
        ratio_PtrToInt_insts       REAL DEFAULT 0.0,
        ratio_Ret_insts            REAL DEFAULT 0.0,
        ratio_SDiv_insts           REAL DEFAULT 0.0,
        ratio_Select_insts         REAL DEFAULT 0.0,
        ratio_SExt_insts           REAL DEFAULT 0.0,
        ratio_Shl_insts            REAL DEFAULT 0.0,
        ratio_ShuffleVector_insts  REAL DEFAULT 0.0,
        ratio_SIToFP_insts         REAL DEFAULT 0.0,
        ratio_SRem_insts           REAL DEFAULT 0.0,
        ratio_Store_insts          REAL DEFAULT 0.0,
        ratio_Sub_insts            REAL DEFAULT 0.0,
        ratio_Switch_insts         REAL DEFAULT 0.0,
        ratio_Trunc_insts          REAL DEFAULT 0.0,
        ratio_UDiv_insts           REAL DEFAULT 0.0,
        ratio_UIToFP_insts         REAL DEFAULT 0.0,
        ratio_URem_insts           REAL DEFAULT 0.0,
        ratio_Xor_insts            REAL DEFAULT 0.0,
        ratio_ZExt_insts           REAL DEFAULT 0.0,
        UNIQUE(sha)
);


CREATE TABLE IF NOT EXISTS OpenCLTidy (
        sha                        TEXT NOT NULL,
        contents                   TEXT NOT NULL,
        UNIQUE(sha)
);
