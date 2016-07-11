-- Generic dataset
CREATE TABLE IF NOT EXISTS Meta (
        key                        TEXT NOT NULL,
        value                      TEXT,
        UNIQUE(key)
);

--
-- Fetched
--

CREATE TABLE IF NOT EXISTS ContentFiles (
        id                  TEXT NOT NULL,       -- Content ID
        contents            TEXT NOT NULL,       -- Contents
        UNIQUE(id)
);


--
-- Preprocessed
--

CREATE TABLE IF NOT EXISTS PreprocessedFiles (
        id                 TEXT NOT NULL,       -- ContentFiles.id
        status             INTEGER NOT NULL,    -- 0 if good, 1 if bad, 2 if ugly
        contents           TEXT NOT NULL,       -- preprocess output
        UNIQUE(id)
);
