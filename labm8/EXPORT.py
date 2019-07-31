from tools.source_tree import export_source_tree
export_source_tree.EXPORT(
    github_repo='labm8',
    targets=['//labm8:all'],
    extra_files=['labm8/labm8.jpg'],
    move_file_mapping={
        'labm8/README.md': 'README.md',
        'labm8/LICENSE': 'LICENSE',
        'labm8/travis.yml': '.travis.yml',
    })
