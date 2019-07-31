from tools.source_tree.deploy import DEPLOY_PIP
DEPLOY_PIP(
    package_name="labm8",
    package_root="//labm8",
    description="Utility libraries for doing science",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords=["utility", "library", "bazel", "protobuf"],
    license="Apache License, Version 2.0",
    long_description_file="//labm8:README.md",
)
