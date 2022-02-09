# -*- coding: utf-8 -*-
# Copyright 2020 the HERA Project
# Licensed under the MIT License

from setuptools import setup

import os
import sys
import json

sys.path.append("hera_notebook_templates")
import version  # noqa

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(os.path.join('hera_notebook_templates', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)


def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths


data_files = package_files('hera_notebook_templates', 'data')

setup_args = {
    'name': 'hera_notebook_templates',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/hera_notebook_templates',
    'license': 'MIT',
    'description': 'Repository for HERA analysis / real-time pipeline notebooks and related support code',
    'package_dir': {'hera_notebook_templates': 'hera_notebook_templates'},
    'packages': ['hera_notebook_templates'],
    'include_package_data': True,
    'scripts': [],
    'version': version.version,
    'package_data': {'hera_notebook_templates': data_files},
    'install_requires': [
        'numpy',
        'matplotlib',
        'pandas',
        'scipy',
        'astropy',
        'pyuvdata',
        'uvtools @ git+git://github.com/HERA-Team/uvtools',
        'hera_mc @ git+git://github.com/HERA-Team/hera_mc',
        'hera_qm @ git+git://github.com/HERA-Team/hera_qm',
        'bokeh',
    ],
    'extras_require': {
        "all": []
    },
    'zip_safe': False,
}


if __name__ == '__main__':
    setup(*(), **setup_args)
