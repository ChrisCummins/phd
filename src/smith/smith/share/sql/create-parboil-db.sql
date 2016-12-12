CREATE TABLE IF NOT EXISTS Meta (
        key                        TEXT NOT NULL,
        value                      TEXT,
        UNIQUE(key)
);

INSERT INTO Meta VALUES("version", "1");
INSERT INTO Meta VALUES("parboil-root", "~/src/parboil");

CREATE TABLE IF NOT EXISTS Benchmarks (
        id                  TEXT NOT NULL,       -- Benchmark ID
        UNIQUE(id)
);

INSERT INTO Benchmarks VALUES("bfs");
INSERT INTO Benchmarks VALUES("cutcp");
INSERT INTO Benchmarks VALUES("histo");
INSERT INTO Benchmarks VALUES("lbm");
INSERT INTO Benchmarks VALUES("mri-gridding");
INSERT INTO Benchmarks VALUES("mri-q");
INSERT INTO Benchmarks VALUES("sad");
INSERT INTO Benchmarks VALUES("sgemm");
INSERT INTO Benchmarks VALUES("spmv");
INSERT INTO Benchmarks VALUES("stencil");
INSERT INTO Benchmarks VALUES("tpacf");

CREATE TABLE IF NOT EXISTS Kernels (
        id                  TEXT NOT NULL,       -- Kernel ID
        benchmark           TEXT NO NULL,        -- Benchmarks.id
        oracle              INT NOT NULL,        -- {0:false, 1:true}
        contents            TEXT NOT NULL,       -- Contents
        UNIQUE(id)
);

CREATE TABLE IF NOT EXISTS Scenarios (
        id                  TEXT NOT NULL,       -- Scenario ID
        host                TEXT NOT NULL,       -- Host ID
        device              TEXT NOT NULL,       -- Device ID
        benchmark           TEXT NOT NULL,       -- Benchmarks.id
        kernel              TEXT NOT NULL,       -- Kernels.id
        dataset             TEXT NOT NULL,       -- Dataset ID
        status              INT NOT NULL,        -- {0:good, 1:bad, 2:unknown}
        UNIQUE(id)
);

CREATE TABLE IF NOT EXISTS Runtimes (
        scenario            TEXT NOT NULL,       -- Scenarios.ID
        io                  REAL NOT NULL,       --
        kernel              REAL NOT NULL,       --
        copy                REAL NOT NULL,       --
        driver              REAL NOT NULL,       --
        compute             REAL NOT NULL,       --
        overlap             REAL NOT NULL,       --
        wall                REAL NOT NULL        --
);
