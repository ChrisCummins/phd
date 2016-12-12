                                smith
                                =====

Fetch and preprocess OpenCL programs from multiple different sources.

Files:

  create-db.sql
        SQL script to create dataset database.

        Usage: sqlite3 <db> < create-db.sql

  create-db-gh.sql
        SQL script to create dataset database for GitHub stream.

        Usage: sqlite3 <db> < create-db-github.sql

  fetch-gh.py
        Download OpenCL files from GitHub.

        Usage: GITHUB_TOKEN=? GITHUB_USERNAME=? GITHUB_PW=? ./fetch-gh.py <db>

  fetch-cs.py
        Generate OpenCL files using clsmith.

        Usage: ./fetch-cs.py <db> [-n <num-kernels>]

  preprocess.py
        Preprocess the fetched data.

        Usage: ./preprocess.py <db>

  explore-gh.py
        GitHub dataset stats.

        Usage: ./explore.py <db>

  train.py
        Train machine learning models on dataset.

        Usage: ./train.py <db>
