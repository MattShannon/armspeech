#!/usr/bin/python
"""A setuptools-based script for distributing and installing armspeech."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.

from setuptools import setup

with open('README.markdown') as readmeFile:
    long_description = readmeFile.read()

requires = [ line.rstrip('\n') for line in open('requirements.txt') ]

setup(
    name = 'armspeech',
    version = '0.6.dev1',
    description = 'Autoregressive probabilistic modelling for speech synthesis.',
    url = 'http://github.com/MattShannon/armspeech',
    author = 'Matt Shannon',
    author_email = 'matt.shannon@cantab.net',
    license = 'various open source licenses (see License file)',
    packages = ['bisque', 'armspeech', 'expt_hts_demo'],
    install_requires = requires,
    # (there are some scripts in the bin directory, but none of them are
    #   suitable for system-wide installation)
    scripts = [],
    long_description = long_description,
)
