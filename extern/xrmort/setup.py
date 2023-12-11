#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
import os

from setuptools import find_packages, setup

from xrmort.versions import NAME, VERSION


def _process_requirements():
    with open('./requirements/requirements.txt') as f:
        packages = f.read().strip().split('\n')
    requires = []
    for pkg in packages:
        if pkg.startswith('git+ssh'):
            return_code = os.system('pip install {}'.format(pkg))
            assert return_code == 0, \
                'error, status_code is: {}, exit!'.format(return_code)
        else:
            requires.append(pkg)
    return requires


def readme():
    with open("./README.md", "r") as f:
        content = f.read()
    return content


setup(
    name=NAME,
    version=VERSION,
    description="XRMoRT",
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='OpenXRLab',
    author_email='openxrlab@pjlab.org.cn',
    packages=find_packages(exclude=('tests')),
    package_data={
        '': ['*.py', 'actors/skeletons/*.json', 'actors/models/*.fbx'],
    },

    python_requires='>=3.7,<3.11',
    install_requires=_process_requirements(),
    extras_require={
        "mathutils": ["mathutils>=2.81,<3.0"],
    },

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    license='Apache License 2.0',
)
