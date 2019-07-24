from tools.source_tree import export_source_tree
export_source_tree.EXPORT(github_repo='dotfiles',
                          targets=[
                              '//system/dotfiles',
                          ],
                          move_file_mapping={
                              'system/dotfiles/README.md': 'README.md',
                          })
