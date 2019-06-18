"""Export a subset of a content files database."""
import sqlalchemy as sql
from datasets.github.scrape_repos import contentfiles

from labm8 import app
from labm8 import humanize

FLAGS = app.FLAGS
app.DEFINE_database(
    'db', contentfiles.ContentFiles,
    'sqlite:////var/phd/experimental/deeplearning/deepsmith/java_fuzz/java.db',
    'URL of the database to modify.')
app.DEFINE_boolean('dry_run', False, 'Whether to save changes.')
app.DEFINE_boolean('reset', False, 'Mark all repositories as active.')
app.DEFINE_boolean('reset_exported', False,
                   'Unmark all repositories as exported.')
app.DEFINE_integer('max_repo_count', None,
                   'Mask by the maximum number of repositories.')
app.DEFINE_integer(
    'min_star_count', 0,
    'Mask by the minimum number of Github stars a repository has.')
app.DEFINE_integer('min_repo_file_count', 0,
                   'Mask by the minimum number of contentfiles in a repo.')
app.DEFINE_integer('max_repo_file_count', 0,
                   'Mask by the maxmium number of contentfiles in a repo.')


def Reset(db: contentfiles.ContentFiles) -> None:
  """Restore active status to database.

  Args:
    db: The database to modify.
  """
  with db.Session(commit=not FLAGS.dry_run) as session:
    inactive_repos = session.query(contentfiles.GitHubRepository)\
      .filter(contentfiles.GitHubRepository.active == False)
    inactive_repos_count = inactive_repos.count()

    repos_count = session.query(contentfiles.GitHubRepository).count()

    app.Log(1, 'Restoring active status to %s of %s repos (%.2f %%)',
            humanize.Commas(inactive_repos_count), humanize.Commas(repos_count),
            (inactive_repos_count / repos_count) * 100)
    inactive_repos.update({"active": True})


def ResetExported(db: contentfiles.ContentFiles) -> None:
  """Restore exported status to database.

  Args:
    db: The database to modify.
  """
  with db.Session(commit=not FLAGS.dry_run) as session:
    exported_repos = session.query(contentfiles.GitHubRepository)\
      .filter(contentfiles.GitHubRepository.exported == True)
    exported_repos_count = exported_repos.count()

    repos_count = session.query(contentfiles.GitHubRepository).count()

    app.Log(1, 'Marking %s of %s repos as not exported (%.2f %%)',
            humanize.Commas(exported_repos_count), humanize.Commas(repos_count),
            (exported_repos_count / repos_count) * 100)
    exported_repos.update({"exported": False})


def MaskOnMaxRepoCount(db: contentfiles.ContentFiles,
                       max_repo_count: int) -> None:
  """Mask by the maximum number of repos.

  Args:
    db: The database to modify.
    max_repo_count: The maximum number of active repos.
  """
  with db.Session(commit=not FLAGS.dry_run) as session:
    active_repos = session.query(contentfiles.GitHubRepository.clone_from_url)\
      .filter(contentfiles.GitHubRepository.active == True)
    active_repos_count = active_repos.count()

    repos_to_mark_inactive_count = max(0, active_repos_count - max_repo_count)

    repos_to_mark_inactive = active_repos\
        .order_by(db.Random())\
        .limit(repos_to_mark_inactive_count)

    app.Log(1, 'Marking %s of %s active repos inactive (%.2f %%)',
            humanize.Commas(repos_to_mark_inactive_count),
            humanize.Commas(active_repos_count),
            (repos_to_mark_inactive_count / active_repos_count) * 100)
    # Can't call Query.update() or Query.delete() when limit() has been called,
    # hence the subquery.
    clone_from_urls = {r[0] for r in repos_to_mark_inactive}
    session.query(contentfiles.GitHubRepository)\
        .filter(contentfiles.GitHubRepository.clone_from_url.in_(clone_from_urls))\
        .update({"active": False}, synchronize_session='fetch')


def MaskOnMinStarCount(db: contentfiles.ContentFiles,
                       min_star_count: int) -> None:
  """Mask by the minimum repo star count.

  Args:
    db: The database to modify.
    min_star_count: The minimum number of stars for a repo to be active.
  """
  with db.Session(commit=not FLAGS.dry_run) as session:
    active_repo_count = session.query(contentfiles.GitHubRepository)\
      .filter(contentfiles.GitHubRepository.active).count()

    repos_to_mark_inactive = session.query(contentfiles.GitHubRepository)\
      .filter(contentfiles.GitHubRepository.active == True)\
      .filter(contentfiles.GitHubRepository.num_stars < min_star_count)
    repos_to_mark_inactive_count = repos_to_mark_inactive.count()

    app.Log(1, 'Marking %s of %s active repos inactive (%.2f %%)',
            humanize.Commas(repos_to_mark_inactive_count),
            humanize.Commas(active_repo_count),
            (repos_to_mark_inactive_count / active_repo_count) * 100)
    repos_to_mark_inactive.update({"active": False})


def MaskOnMinRepoFileCount(db: contentfiles.ContentFiles,
                           min_repo_file_count: int) -> None:
  """Mask by the minimum repo file count.

  Args:
    db: The database to modify.
    min_repo_file_count: The minimum number of contentfiles in a repo for it to
        be active.
  """
  with db.Session(commit=not FLAGS.dry_run) as session:
    active_repo_count = session.query(contentfiles.GitHubRepository)\
      .filter(contentfiles.GitHubRepository.active).count()

    repos_to_mark_inactive = session.query(
          contentfiles.ContentFile.clone_from_url,
          sql.func.count(contentfiles.ContentFile.clone_from_url))\
      .join(contentfiles.GitHubRepository)\
      .filter(contentfiles.GitHubRepository.active == True)\
      .group_by(contentfiles.ContentFile.clone_from_url) \
      .having(sql.func.count(contentfiles.ContentFile.clone_from_url) <
              min_repo_file_count)
    repos_to_mark_inactive_count = repos_to_mark_inactive.count()

    app.Log(1, 'Marking %s of %s active repos inactive (%.2f %%)',
            humanize.Commas(repos_to_mark_inactive_count),
            humanize.Commas(active_repo_count),
            (repos_to_mark_inactive_count / active_repo_count) * 100)

    # Can't call Query.update() or Query.delete() when limit() has been called,
    # hence the subquery.
    clone_from_urls = {r.clone_from_url for r in repos_to_mark_inactive}
    session.query(contentfiles.GitHubRepository)\
        .filter(contentfiles.GitHubRepository.clone_from_url.in_(clone_from_urls))\
        .update({"active": False}, synchronize_session='fetch')


def MaskOnMaxRepoFileCount(db: contentfiles.ContentFiles,
                           max_repo_file_count: int) -> None:
  """Mask by the maximum repo file count.

  Args:
    db: The database to modify.
    max_repo_file_count: The maxmium number of contentfiles in a repo for it to
        be active.
  """
  with db.Session(commit=not FLAGS.dry_run) as session:
    active_repo_count = session.query(contentfiles.GitHubRepository)\
      .filter(contentfiles.GitHubRepository.active).count()

    repos_to_mark_inactive = session.query(
          contentfiles.ContentFile.clone_from_url,
          sql.func.count(contentfiles.ContentFile.clone_from_url))\
      .join(contentfiles.GitHubRepository)\
      .filter(contentfiles.GitHubRepository.active == True)\
      .group_by(contentfiles.ContentFile.clone_from_url) \
      .having(sql.func.count(contentfiles.ContentFile.clone_from_url) >
              max_repo_file_count)
    repos_to_mark_inactive_count = repos_to_mark_inactive.count()

    app.Log(1, 'Marking %s of %s active repos inactive (%.2f %%)',
            humanize.Commas(repos_to_mark_inactive_count),
            humanize.Commas(active_repo_count),
            (repos_to_mark_inactive_count / active_repo_count) * 100)

    # Can't call Query.update() or Query.delete() when limit() has been called,
    # hence the subquery.
    clone_from_urls = {r.clone_from_url for r in repos_to_mark_inactive}
    session.query(contentfiles.GitHubRepository)\
        .filter(contentfiles.GitHubRepository.clone_from_url.in_(clone_from_urls))\
        .update({"active": False}, synchronize_session='fetch')


def main():
  """Main entry point."""

  db = FLAGS.db()
  if FLAGS.reset and FLAGS.reset_exported:
    Reset(db)
    ResetExported(db)
  elif FLAGS.reset:
    Reset(db)
  elif FLAGS.reset_exported:
    ResetExported(db)
  elif FLAGS.max_repo_count:
    MaskOnMaxRepoCount(db, FLAGS.max_repo_count)
  elif FLAGS.min_star_count:
    MaskOnMinStarCount(db, FLAGS.min_star_count)
  elif FLAGS.min_repo_file_count:
    MaskOnMinRepoFileCount(db, FLAGS.min_repo_file_count)
  elif FLAGS.max_repo_file_count:
    MaskOnMaxRepoFileCount(db, FLAGS.max_repo_file_count)
  if FLAGS.dry_run:
    app.Log(1, 'Dry run, rolling back ...')


if __name__ == '__main__':
  app.Run(main)
