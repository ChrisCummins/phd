                                  ml
                                  ==

Files:

  create-db.sql
        SQL script to create dataset database.

        Usage: sqlite3 dataset.db < create-db.sql

  fetch.py
        Download all the OpenCL on GitHub.

        Usage: GITHUB_TOKEN=? GITHUB_USERNAME=? GITHUB_PW=? ./fetch.py <db>

  preprocess.py
        Preprocess the fetched data.

        Usage: ./preprocess.py <db>

  explore.py
        Perform initial dataset exploration.

        Usage: ./explore.py <db>

  train.py
        Train machine learning models on dataset.

        Usage: ./train.py <db>
