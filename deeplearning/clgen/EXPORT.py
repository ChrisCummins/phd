from tools.source_tree import export_source_tree
export_source_tree.EXPORT(
    github_repo='clgen',
    targets=['//deeplearning/clgen', 'tests(//deeplearning/clgen/...)'],
    move_file_mapping={
        'deeplearning/clgen/README.md': 'README.md',
        'deeplearning/clgen/LICENSE': 'LICENSE',
        'deeplearning/clgen/travis.yml': '.travis.yml',
    })
