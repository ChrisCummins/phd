"""Export a subset of a content files database."""

from datasets.github.scrape_repos import contentfiles

from labm8 import app
from labm8 import humanize

FLAGS = app.FLAGS
app.DEFINE_integer(
    'n', 100, 'The number of repositories to export the content files of.')
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


def main():
  """Main entry point."""

  input_db = FLAGS.input()
  output_db = FLAGS.output()

  with input_db.Session() as ins, output_db.Session(commit=True) as outs:
    repo_count = ins.query(contentfiles.GitHubRepository).count()
    export_count = min(FLAGS.n, repo_count)

    app.Log(1, 'Exporting content files from %s of %s repos',
            humanize.Commas(export_count), humanize.Commas(repo_count))
    repos_to_export = ins.query(contentfiles.GitHubRepository)\
        .order_by(input_db.Random()).limit(export_count)
    ImportQueryResults(repos_to_export, outs)

    clone_from_urls = {t.clone_from_url for t in repos_to_export}

    contentfiles_to_export = ins.query(contentfiles.ContentFile)\
      .filter(contentfiles.ContentFile.clone_from_url.in_(clone_from_urls))
    app.Log(1, 'Exporting %s content files',
            humanize.Commas(contentfiles_to_export.count()))
    ImportQueryResults(contentfiles_to_export, outs)


if __name__ == '__main__':
  app.Run(main)
