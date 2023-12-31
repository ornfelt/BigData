#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "matplotlib",
    "pandas",
    "textmining3",
    "wordcloud",
    "scikit-learn",
    "nltk",
    "tqdm",
    "gensim",
    "pyLDAvis",
    "langdetect",
    "seaborn"
]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="David Johnson",
    author_email='david.johnson@im.uu.se',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',        
    ],
    description="Wrapper library for Sociala Medier och Digitala Metoder labs",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='somialabs',
    name='somialabs',
    packages=find_packages(include=['somialabs']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/UppsalaIM/2IS060/somialabs',
    version='0.6.0',
    zip_safe=False,
)
