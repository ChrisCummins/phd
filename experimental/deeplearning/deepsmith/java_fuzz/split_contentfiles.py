"""Export a subset of a content files database."""

from datasets.github.scrape_repos import contentfiles

from labm8 import app
from labm8 import humanize

FLAGS = app.FLAGS
app.DEFINE_integer(
    'max_repo_count', None,
    'The number of repositories to export the content files of.')
app.DEFINE_integer('min_star_count', 0,
                   'The minimum number of stars for a repo to be exported.')
app.DEFINE_database(
    'input', contentfiles.ContentFiles,
    'sqlite:////var/phd/experimental/deeplearning/deepsmith/java_fuzz/java.db',
    'URL of the database to export content files from.')
app.DEFINE_database(
    'output', contentfiles.ContentFiles,
    'sqlite:////tmp/phd/experimental/deeplearning/deepsmith/java_fuzz/java_subset.db',
    'URL of the database to export content files to.')


def ImportQueryResults(query, session):
  """Copy results of a query from one session into a new session."""
  # You can't simply use session.add_all() when the objects are already attached
  # to a different session.
  for row in query:
    session.merge(row)


def SplitContentFiles(input_db: contentfiles.ContentFiles,
                      output_db: contentfiles.ContentFiles,
                      minimum_star_count: int, max_repo_count: int) -> None:
  """Split a content files database.

  Args:
    input_db: The database to export from.
    output_db: The database to export to.
    minimum_star_count: The minimum number of stars for a repo to export.
    max_repo_count: The maximum number of repos to export.
  """
  with input_db.Session() as ins, output_db.Session(commit=True) as outs:
    total_repo_count = ins.query(contentfiles.GitHubRepository).count()
    max_repo_count = max_repo_count or total_repo_count

    candidate_repos = ins.query(contentfiles.GitHubRepository) \
        .filter(contentfiles.GitHubRepository.num_stars >= minimum_star_count)

    repo_export_count = min(max_repo_count, candidate_repos.count())

    app.Log(1, 'Exporting content files from %s of %s repos',
            humanize.Commas(repo_export_count),
            humanize.Commas(total_repo_count))

    repos_to_export = candidate_repos\
        .order_by(input_db.Random())\
        .limit(max_repo_count)

    ImportQueryResults(repos_to_export, outs)

    clone_from_urls = {t.clone_from_url for t in repos_to_export}

    contentfiles_to_export = ins.query(contentfiles.ContentFile) \
      .filter(contentfiles.ContentFile.clone_from_url.in_(clone_from_urls))
    app.Log(1, 'Exporting %s content files',
            humanize.Commas(contentfiles_to_export.count()))
    ImportQueryResults(contentfiles_to_export, outs)


def main():
  """Main entry point."""

  input_db = FLAGS.input()
  output_db = FLAGS.output()

  SplitContentFiles(input_db, output_db, FLAGS.minimum_star_count,
                    FLAGS.max_repo_count)


if __name__ == '__main__':
  app.Run(main)
