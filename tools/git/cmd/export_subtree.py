"""Export a subset of a git repo."""
import git

from labm8.py import app
from tools.git import export_subtree

app.DEFINE_input_path(
  "source", None, "The source repository.", required=True, is_dir=True
)
app.DEFINE_output_path(
  "destination",
  None,
  "The path of the destination repository. If this path does not exist, an "
  "empty repository is created.",
  required=True,
  is_dir=True,
)
app.DEFINE_list("files", None, "A list of files to export.", required=True)
app.DEFINE_string("head", "HEAD", "The head reference.")

FLAGS = app.FLAGS


def Main():
  """Main entry point."""
  # Get or create the destination repo.
  if (FLAGS.destination / ".git").is_dir():
    destination = git.Repo(FLAGS.destination)
  else:
    FLAGS.destination.mkdir(parents=True, exist_ok=True)
    destination = git.Repo.init(FLAGS.destination)

  export_subtree.ExportSubtree(
    git.Repo(FLAGS.source),
    destination,
    files_of_interest=set(FLAGS.files),
    head_ref=FLAGS.head,
  )


if __name__ == "__main__":
  app.Run(Main)
