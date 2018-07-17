# Docs

## Adding a new document

1. Create `//docs/YY_MM_NAME:YY_MM_NAME` build target.
1. Add an entry to Publications or Misc section of `//README.txt`.
1. (If necessary) Add an entry to publications section of `//http/chriscummins.cc/cv.json`.
1. (If necessary) Add an entry to publications section of `//docs/cv/sec/publications.tex`.
1. (If necessary) Build new CV: `bazel build //docs/cv`.
1. (If necessary) Update CV file: `cp ~/phd/bazel-genfiles/docs/cv/cv.pdf ~/phd/http/chriscummins/cv.pdf`.
1. (If necessary) Commit and bump changes in `//http/chriscummins.cc`.
1. Commit changes.
