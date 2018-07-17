# Talks

## Adding a new talk

1. Create `//talks/YY_MM_NAME/YY_MM_NAME` folder.
1. Add PDF of slides to `//talks/YY_MM_NAME/YY_MM_NAME.pdf`.
1. Move slides sources to `//talks/YY_MM_NAME/YY_MM_NAME/YY_MM_NAME.key`.
1. Move assets to `//talks/YY_MM_NAME/YY_MM_NAME/assets/`.
1. Create dpack manifest: `$ bazel run //lib/dpack -- --package ~/phd/talks/YY_MM_NAME/YY_MM_NAME --init`.
1. Add comments to manifest file.
1. Create dpack archive: `$ bazel run //lib/dpack -- --package ~/phd/talks/YY_MM_NAME/YY_MM_NAME`.
1. Add `/talks/YY_MM_NAME/YY_MM_NAME/` to `~/phd/.gitignore`.
1. Add an entry to Talks section of `//README.txt`.
1. (If necessary) Add an entry to talks section of `//http/chriscummins.cc/cv.json`.
1. (If necessary) Add an entry to talks section of `//docs/cv/sec/invited_talks.tex`.
1. (If necessary) Build new CV: `bazel build //docs/cv`.
1. (If necessary) Update CV file: `cp ~/phd/bazel-genfiles/docs/cv/cv.pdf ~/phd/http/chriscummins/cv.pdf`.
1. (If necessary) Commit and bump changes in `//http/chriscummins.cc`.
1. Commit changes.
