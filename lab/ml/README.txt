                                  ml
                                  ==

Files:

  create-db.sql
        SQL script to create dataset database.

        Usage: sqlite3 dataset.db < create-db.sql

  fetch.py
        Download all the OpenCL on GitHub.

        Usage: GITHUB_TOKEN=? GITHUB_USERNAME=? GITHUB_PW=? ./fetch.py <db>

  fetch-clsmith.py
        Generate kernels using clsmith.

        Usage: ./fetch-clsmith.py <db> [-n <num-kernels>]

  preprocess.py
        Preprocess the fetched data.

        Usage: ./preprocess.py <db>

  preprocess-clsmith.py
        Preprocess the fetched data (clsmith kernels).

        Usage: ./preprocess-clsmith.py <db>

  explore.py
        Perform initial dataset exploration.

        Usage: ./explore.py <db>

  train.py
        Train machine learning models on dataset.

        Usage: ./train.py <db>
