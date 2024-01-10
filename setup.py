#!/usr/bin/python
# Credit to Parth Nobel for this file
import subprocess
from setuptools import setup, find_packages

# get all the git tags from the cmd line that follow our versioning pattern
git_tags = subprocess.Popen(
    ["git", "tag", "--list", "v*[0-9]", "--sort=version:refname"],
    stdout=subprocess.PIPE,
)
tags = git_tags.stdout.read()
git_tags.stdout.close()
tags = tags.decode("utf-8").split("\n")
tags.sort()

# PEP 440 won't accept the v in front, so here we remove it, strip the new line and decode the byte stream
VERSION_FROM_GIT_TAG = "1.0.0" #tags[-1][1:]

setup(
    name="ls_spa",
    version=VERSION_FROM_GIT_TAG,  # Required
    setup_requires=["setuptools>=18.0"],
    packages=find_packages(exclude=["notebooks"]),  # Required
    install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'matplotlib',
      ],
    description="Efficient Shapley performance attribution for least-squares problems",
    url="https://github.com/cvxgrp/ls-spa",
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
