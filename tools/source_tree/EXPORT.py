from tools.source_tree import export_source_tree
export_source_tree.EXPORT(github_repo='bazel_subtree_github_export',
                          targets=['//tools/source_tree:export_source_tree'],
                          move_file_mapping={
                              'tools/source_tree/README.md': 'README.md',
                              'tools/source_tree/LICENSE': 'LICENSE',
                              'tools/source_tree/travis.yml': '.travis.yml',
                          })
