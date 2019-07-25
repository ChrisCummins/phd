from tools.source_tree import export_source_tree
export_source_tree.EXPORT(
    github_repo='jasper',
    targets=[
        '//util/jasper/...',
    ],
    move_file_mapping={
        'util/jasper/README.md': 'README.md',
        'util/jasper/LICENSE': 'LICENSE',
    })
