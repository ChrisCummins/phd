CREATE TABLE IF NOT EXISTS Meta (
        key                        TEXT NOT NULL,
        value                      TEXT,
        UNIQUE(key)
);

--
-- Fetched
--

CREATE TABLE IF NOT EXISTS ContentFiles (
        sha                 TEXT NOT NULL,       -- File checksum
        contents            TEXT NOT NULL,       -- File contents
        UNIQUE(sha)
);

--
-- Preprocessed
--

CREATE TABLE IF NOT EXISTS PreprocessedFiles (
        sha                TEXT NOT NULL,       -- ContentFiles.sha
        status             INTEGER NOT NULL,    -- 0 if good, 1 if bad, 2 if ugly
        contents           TEXT NOT NULL,       -- preprocess output
        UNIQUE(sha)
);
