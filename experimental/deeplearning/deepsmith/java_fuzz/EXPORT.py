from tools.source_tree import export_source_tree
export_source_tree.EXPORT(
    github_repo='ibm_deepsmith_shared_code',
    targets=[
        '//deeplearning/clgen/preprocessors:JavaRewriter',
        '//deeplearning/deepsmith/harnesses:JavaDriver',
        '//deeplearning/deepsmith/harnesses:JavaDriverTest',
        '//deeplearning/clgen/preprocessors:java_test',
    ],
    copy_file_mapping={
        '.gitignore': '.gitignore',
    },
    move_file_mapping={
        'experimental/deeplearning/deepsmith/java_fuzz/shared_code_README.md':
        'README.md',
    })
