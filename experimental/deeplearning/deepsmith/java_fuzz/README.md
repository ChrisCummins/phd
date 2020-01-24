# Extending DeepSmith to Java

This is the home for work on extending DeepSmith to fuzzing JVM implementations.

## Usage

### Scraping Java files from Github

1. Create a [personal access token](https://github.com/settings/tokens) on
   Github and make a note of it. When creating the access token, Github will
   present a bunch of checkboxes for permissions. Don't tick any of them.
2. Launch/find a MySQL server to store the database of scraped files.
3. Run the docker scraper image, providing the access token and address of the
   MySQL server:
```
$ docker run --memory=4g chriscummins/java_fuzz_scraper:latest \
    --github_access_token=<access_token> \
    --db='mysql://<user>:<pass>@<host>/<database_name>?charset=utf8'
```

Alternatively, SQLite or PostgreSQL can be used in place of MySQL, see [here](https://github.com/ChrisCummins/phd/blob/1217c228cb9c0c37e3b85670052d9ca51cd74a2b/labm8/sqlutil.py#L89-L115) for the URL scheme used by the `--db` parameter of the docker image. If writing to an SQLite database, be sure to [share the volume](https://docs.docker.com/storage/volumes/#choose-the--v-or---mount-flag) with the host.


### Exporting a subset of scraped files

It can be useful for development and debugging to work with a small database of
scraped files. The `java_fuzz_split_contentfiles` image will duplicate
content files from a random subset of scraped repositories. E.g. to duplicate
the content files from 10 random repos with a minimum Github star count of 5
in a MySQL database to an SQLite database `/tmp/fuzz/java_subset.db`:

```sh
$ docker run --memory=4g -v/tmp/fuzz:/tmp/fuzz \
    chriscummins/java_fuzz_split_contentfiles:latest \
    --input='mysql://<user>:<pass>@<host>/<database_name>?charset=utf8' \
    --output='sqlite:////tmp/fuzz/java_subset.db' --max_repo_count=10 \
    --min_star_count=5
```

### Exporting a corpus of scraped Java files

The `export_java_corpus` image is used to run preprocessing functions on the
scraped Java files and export them to a directory:

```sh
$ docker run --memory=12g chriscummins/export_java_corpus:latest \
    --db='mysql://<user>:<pass>@<host>/<database_name>?charset=utf8' \
    --outdir=/tmp/fuzz/corpus \
    --preprocessors="datasets.github.scrape_repos.preprocessors.extractors:JavaStaticMethods"
```
