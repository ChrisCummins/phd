from tools.source_tree import export_source_tree
export_source_tree.EXPORT(github_repo='cldrive',
                          targets=[
                              '//gpu/cldrive/...',
                          ],
                          move_file_mapping={
                              'gpu/cldrive/README.md': 'README.md',
                              'gpu/cldrive/LICENSE': 'LICENSE',
                              'gpu/cldrive/travis.yml': '.travis.yml',
                          })
