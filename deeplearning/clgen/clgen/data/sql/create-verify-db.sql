CREATE TABLE IF NOT EXISTS Meta (
        key                     TEXT NOT NULL,
        value                   TEXT,
        UNIQUE(key)
);

CREATE TABLE IF NOT EXISTS Data (
        id                      TEXT NOT NULL,
        status                  INTEGER NOT NULL,
        result                  TEXT,
        UNIQUE(id)
);

CREATE TABLE IF NOT EXISTS Source (
        id                      TEXT NOT NULL,
        source                  TEXT NOT NULL,
        UNIQUE(id)
);
